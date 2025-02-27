"""
This module contains classes and functions that are used for FCOS, a  one-stage 
object detector. You have to implement the functions here -
walk through the notebooks and you will find instructions on *when* to implement
*what* in this module.
"""


import math
from typing import Dict, List, Optional

import torch
from tools.loading import *
from torch import nn
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import torchvision
from torchvision.models import feature_extraction
from torchvision.ops import sigmoid_focal_loss


def hello_fcos():
    print("Hello from fcos.py!")


class DetectorBackboneWithFPN(nn.Module):
    r"""
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights for faster convergence.
        _cnn = models.regnet_x_400mf(pretrained=True)

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        self.fpn_params = nn.ModuleDict()

        #print(dummy_out_shapes[0][1][1])
        self.fpn_params['l3'] = nn.Conv2d(dummy_out_shapes[0][1][1], self.out_channels, 1, 1, 0) # lateral layers added to dictionary
        self.fpn_params['l4'] = nn.Conv2d(dummy_out_shapes[1][1][1], self.out_channels, 1, 1, 0)
        self.fpn_params['l5'] = nn.Conv2d(dummy_out_shapes[2][1][1], self.out_channels, 1, 1, 0)
        self.fpn_params['p3'] = nn.Conv2d(self.out_channels,self.out_channels,3,1,1) # output layers
        self.fpn_params['p4'] = nn.Conv2d(self.out_channels,self.out_channels,3,1,1)
        self.fpn_params['p5'] = nn.Conv2d(self.out_channels,self.out_channels,3,1,1)

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}

        # Replace "PASS" statement with your code
        lat3 = self.fpn_params['l3'](backbone_feats['c3'])
        lat4 = self.fpn_params['l4'](backbone_feats['c4'])
        lat5 = self.fpn_params['l5'](backbone_feats['c5'])

        lat5_resize = nn.functional.interpolate(lat5, size=(lat4.shape[2],lat4.shape[3]), mode='nearest')
        lat4 = lat4 + lat5_resize
        lat4_resize = nn.functional.interpolate(lat4, size=(lat3.shape[2],lat3.shape[3]), mode='nearest')
        lat3 = lat3 + lat4_resize

        fpn_feats['p3'] = self.fpn_params['p3'](lat3)
        fpn_feats['p4'] = self.fpn_params['p4'](lat4)
        fpn_feats['p5'] = self.fpn_params['p5'](lat5)

        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `fcos.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    """

    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        # Get feature map shape
        _, _, H, W = feat_shape
        # Create grid of indices
        y_grid, x_grid = torch.meshgrid(
            torch.arange(0, H, device=device, dtype=dtype),
            torch.arange(0, W, device=device, dtype=dtype),
        )
        # Compute absolute pixel locations
        x_coords = (x_grid + 0.5) * level_stride
        y_coords = (y_grid + 0.5) * level_stride
        # Stack coordinates along the last dimension
        coords = torch.stack((x_coords, y_coords), dim=-1)
        # Reshape to (H * W, 2) and assign to location_coords dictionary
        location_coords[level_name] = coords.reshape(-1, 2)
    return location_coords


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    # Sort boxes by scores in descending order
    _, idxs = scores.sort(descending=True)
    keep = []

    while idxs.numel() > 0:
        # Add the current top index to 'keep' list
        current = idxs[0]
        keep.append(current.item())

        if idxs.numel() == 1:
            break

        # Compute IoU of the remaining boxes with the top box
        current_box = boxes[current, None]
        remaining_boxes = boxes[idxs[1:], :]
        ious = compute_iou(current_box, remaining_boxes).squeeze()

        # Keep only the boxes with IoU less than the threshold
        idxs = idxs[1:][ious < iou_threshold]

    return torch.tensor(keep, dtype=torch.long)

def compute_iou(box1, box2):
    # Compute the intersection over union of two sets of boxes.
    # The boxes are expected in (x1, y1, x2, y2) format.

    inter_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_y2 = torch.min(box1[..., 3], box2[..., 3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area


def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


# Short hand type notation:
TensorDict = Dict[str, torch.Tensor]





class FCOSPredictionNetwork(nn.Module):
    """
    FCOS prediction network that accepts FPN feature maps from different levels
    and makes three predictions at every location: bounding boxes, class ID and
    centerness. This module contains a "stem" of convolution layers, along with
    one final layer per prediction. For a visual depiction, see Figure 2 (right
    side) in FCOS paper: https://arxiv.org/abs/1904.01355

    We will use feature maps from FPN levels (P3, P4, P5) and exclude (P6, P7).
    """

    def __init__(
        self, num_classes: int, in_channels: int, stem_channels: List[int]
    ):
        """
        Args:
            num_classes: Number of object classes for classification.
            in_channels: Number of channels in input feature maps. This value
                is same as the output channels of FPN, since the head directly
                operates on them.
            stem_channels: List of integers giving the number of output channels
                in each convolution layer of stem layers.
        """
        super().__init__()

        # Fill these.
        stem_cls = []
        stem_box = []
        # Replace "PASS" statement with your code
        # Determine the number of convolution layers in the stem
        num_layers = len(stem_channels)
        # Add alternating Conv2d and ReLU layers to each stem
        for i in range(num_layers - 1):
            # Convolution layer
            conv_layer_cls = nn.Conv2d(
                in_channels if i == 0 else stem_channels[i - 1],
                stem_channels[i],
                kernel_size=3,
                stride=1,
                padding=1,
            )
            conv_layer_box = nn.Conv2d(
                in_channels if i == 0 else stem_channels[i - 1],
                stem_channels[i],
                kernel_size=3,
                stride=1,
                padding=1,
            )
            # Initialize weights from a normal distribution
            nn.init.normal_(conv_layer_cls.weight, mean=0, std=0.01)
            nn.init.constant_(conv_layer_cls.bias, 0)
            nn.init.normal_(conv_layer_box.weight, mean=0, std=0.01)
            nn.init.constant_(conv_layer_box.bias, 0)
            # Add to respective stems
            stem_cls.extend([conv_layer_cls, nn.ReLU(inplace=True)])
            stem_box.extend([conv_layer_box, nn.ReLU(inplace=True)])
        # Final layer without ReLU activation
        final_conv_cls = nn.Conv2d(
            stem_channels[-2], stem_channels[-1], kernel_size=3, stride=1, padding=1
        )
        final_conv_box = nn.Conv2d(
            stem_channels[-2], stem_channels[-1], kernel_size=3, stride=1, padding=1
        )
        # Initialize weights from a normal distribution
        nn.init.normal_(final_conv_cls.weight, mean=0, std=0.01)
        nn.init.constant_(final_conv_cls.bias, 0)
        nn.init.normal_(final_conv_box.weight, mean=0, std=0.01)
        nn.init.constant_(final_conv_box.bias, 0)
        # Add final layers to respective stems
        stem_cls.append(final_conv_cls)
        stem_box.append(final_conv_box)
        # Wrap the layers defined by student into a `nn.Sequential` module:
        self.stem_cls = nn.Sequential(*stem_cls)
        self.stem_box = nn.Sequential(*stem_box)


        # Wrap the layers defined by student into a `nn.Sequential` module:
        self.stem_cls = nn.Sequential(*stem_cls)
        self.stem_box = nn.Sequential(*stem_box)

        self.pred_cls = None  # Class prediction conv
        self.pred_box = None  # Box regression conv
        self.pred_ctr = None  # Centerness conv

        # Replace "pass" statement with your code
        self.pred_cls = nn.Conv2d(
            stem_channels[-1], num_classes, kernel_size=3, stride=1, padding=1
        )
        self.pred_box = nn.Conv2d(
            stem_channels[-1], 4, kernel_size=3, stride=1, padding=1
        )
        self.pred_ctr = nn.Conv2d(
            stem_channels[-1], 1, kernel_size=3, stride=1, padding=1
        )


        # OVERRIDE: Use a negative bias in `pred_cls` to improve training
        # stability. Without this, the training will most likely diverge.
        # STUDENTS: You do not need to get into details of why this is needed.
        torch.nn.init.constant_(self.pred_cls.bias, -math.log(99))

    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        """
        Accept FPN feature maps and predict the desired outputs at every location
        (as described above). Format them such that channels are placed at the
        last dimension, and (H, W) are flattened (having channels at last is
        convenient for computing loss as well as perforning inference).

        Args:
            feats_per_fpn_level: Features from FPN, keys {"p3", "p4", "p5"}. Each
                tensor will have shape `(batch_size, fpn_channels, H, W)`. For an
                input (224, 224) image, H = W are (28, 14, 7) for (p3, p4, p5).

        Returns:
            List of dictionaries, each having keys {"p3", "p4", "p5"}:
            1. Classification logits: `(batch_size, H * W, num_classes)`.
            2. Box regression deltas: `(batch_size, H * W, 4)`
            3. Centerness logits:     `(batch_size, H * W, 1)`
        """

        class_logits = {}
        boxreg_deltas = {}
        centerness_logits = {}

        for level, feature in feats_per_fpn_level.items():
                    # Generating class logits
                    class_logits[level] = self.pred_cls(self.stem_cls(feature))
                    batch_size = class_logits[level].shape[0]
                    num_classes = class_logits[level].shape[1]
                    class_logits[level] = class_logits[level].view(batch_size, num_classes, -1).permute(0, 2, 1)

                    # Processing bounding box regression deltas
                    stem_output = self.stem_box(feature)
                    boxreg_deltas[level] = self.pred_box(stem_output)
                    boxreg_deltas[level] = boxreg_deltas[level].view(batch_size, 4, -1).permute(0, 2, 1)

                    # Calculating centerness logits
                    centerness_logits[level] = self.pred_ctr(stem_output)
                    centerness_logits[level] = centerness_logits[level].view(batch_size, 1, -1).permute(0, 2, 1)

        return [class_logits, boxreg_deltas, centerness_logits]


@torch.no_grad()
def fcos_match_locations_to_gt(
    locations_per_fpn_level: TensorDict,
    strides_per_fpn_level: Dict[str, int],
    gt_boxes: torch.Tensor,
) -> TensorDict:
    """
    Match centers of the locations of FPN feature with a set of GT bounding
    boxes of the input image. Since our model makes predictions at every FPN
    feature map location, we must supervise it with an appropriate GT box.
    There are multiple GT boxes in image, so FCOS has a set of heuristics to
    assign centers with GT, which we implement here.

    NOTE: This function is NOT BATCHED. Call separately for GT box batches.

    Args:
        locations_per_fpn_level: Centers at different levels of FPN (p3, p4, p5),
            that are already projected to absolute co-ordinates in input image
            dimension. Dictionary of three keys: (p3, p4, p5) giving tensors of
            shape `(H * W, 2)` where H = W is the size of feature map.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `fcos.py` for more details.
        gt_boxes: GT boxes of a single image, a batch of `(M, 5)` boxes with
            absolute co-ordinates and class ID `(x1, y1, x2, y2, C)`. In this
            codebase, this tensor is directly served by the dataloader.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(N, 5)` GT boxes, one for each center. They are
            one of M input boxes, or a dummy box called "background" that is
            `(-1, -1, -1, -1, -1)`. Background indicates that the center does
            not belong to any object.
    """

    matched_gt_boxes = {
        level_name: None for level_name in locations_per_fpn_level.keys()
    }

    # Do this matching individually per FPN level.
    for level_name, centers in locations_per_fpn_level.items():

        # Get stride for this FPN level.
        stride = strides_per_fpn_level[level_name]

        x, y = centers.unsqueeze(dim=2).unbind(dim=1)
        x0, y0, x1, y1 = gt_boxes[:, :4].unsqueeze(dim=0).unbind(dim=2)
        pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)

        # Pairwise distance between every feature center and GT box edges:
        # shape: (num_gt_boxes, num_centers_this_level, 4)
        pairwise_dist = pairwise_dist.permute(1, 0, 2)

        # The original FCOS anchor matching rule: anchor point must be inside GT.
        match_matrix = pairwise_dist.min(dim=2).values > 0

        # Multilevel anchor matching in FCOS: each anchor is only responsible
        # for certain scale range.
        # Decide upper and lower bounds of limiting targets.
        pairwise_dist = pairwise_dist.max(dim=2).values

        lower_bound = stride * 4 if level_name != "p3" else 0
        upper_bound = stride * 8 if level_name != "p5" else float("inf")
        match_matrix &= (pairwise_dist > lower_bound) & (
            pairwise_dist < upper_bound
        )

        # Match the GT box with minimum area, if there are multiple GT matches.
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (
            gt_boxes[:, 3] - gt_boxes[:, 1]
        )

        # Get matches and their labels using match quality matrix.
        match_matrix = match_matrix.to(torch.float32)
        match_matrix *= 1e8 - gt_areas[:, None]

        # Find matched ground-truth instance per anchor (un-matched = -1).
        match_quality, matched_idxs = match_matrix.max(dim=0)
        matched_idxs[match_quality < 1e-5] = -1

        # Anchors with label 0 are treated as background.
        matched_boxes_this_level = gt_boxes[matched_idxs.clip(min=0)]
        matched_boxes_this_level[matched_idxs < 0, :] = -1

        matched_gt_boxes[level_name] = matched_boxes_this_level

    return matched_gt_boxes


def fcos_get_deltas_from_locations(
    locations: torch.Tensor, gt_boxes: torch.Tensor, stride: int
) -> torch.Tensor:
    """
    Compute distances from feature locations to GT box edges. These distances
    are called "deltas" - `(left, top, right, bottom)` or simply `LTRB`. The
    feature locations and GT boxes are given in absolute image co-ordinates.

    These deltas are used as targets for training FCOS to perform box regression
    and centerness regression. They must be "normalized" by the stride of FPN
    feature map (from which feature locations were computed, see the function
    `get_fpn_location_coords`). If GT boxes are "background", then deltas must
    be `(-1, -1, -1, -1)`.

    NOTE: This transformation function should not require GT class label. Your
    implementation must work for GT boxes being `(N, 4)` or `(N, 5)` tensors -
    without or with class labels respectively. You may assume that all the
    background boxes will be `(-1, -1, -1, -1)` or `(-1, -1, -1, -1, -1)`.

    Args:
        locations: Tensor of shape `(N, 2)` giving `(xc, yc)` feature locations.
        gt_boxes: Tensor of shape `(N, 4 or 5)` giving GT boxes.
        stride: Stride of the FPN feature map.

    Returns:
        torch.Tensor
            Tensor of shape `(N, 4)` giving deltas from feature locations, that
            are normalized by feature stride.
    """
    deltas = None

    deltas = torch.empty(gt_boxes.shape[0],4).to(device=gt_boxes.device)

    deltas[:,3] = (gt_boxes[:,3]-locations[:,1]) / stride
    deltas[:,2] = (gt_boxes[:,2]-locations[:,0]) / stride
    deltas[:,1] = (locations[:,1]-gt_boxes[:,1]) / stride
    deltas[:,0] = (locations[:,0]-gt_boxes[:,0]) / stride

    deltas[gt_boxes[:,:4].sum(dim=1)==-4]=-1

    return deltas


def fcos_apply_deltas_to_locations(
    deltas: torch.Tensor, locations: torch.Tensor, stride: int
) -> torch.Tensor:
    """
    Implement the inverse of `fcos_get_deltas_from_locations` here:

    Given edge deltas (left, top, right, bottom) and feature locations of FPN, get
    the resulting bounding box co-ordinates by applying deltas on locations. This
    method is used for inference in FCOS: deltas are outputs from model, and
    applying them to anchors will give us final box predictions.

    Recall in above method, we were required to normalize the deltas by feature
    stride. Similarly, we have to un-normalize the input deltas with feature
    stride before applying them to locations, because the given input locations are
    already absolute co-ordinates in image dimensions.

    Args:
        deltas: Tensor of shape `(N, 4)` giving edge deltas to apply to locations.
        locations: Locations to apply deltas on. shape: `(N, 2)`
        stride: Stride of the FPN feature map.

    Returns:
        torch.Tensor
            Same shape as deltas and locations, giving co-ordinates of the
            resulting boxes `(x1, y1, x2, y2)`, absolute in image dimensions.
    """
    output_boxes = None
    deltas = deltas.clip(min=0)
    output_boxes = torch.empty(deltas.size()).to(device=deltas.device)

    output_boxes[:,0]=locations[:,0]-stride*deltas[:,0]
    output_boxes[:,1]=locations[:,1]-stride*deltas[:,1]
    output_boxes[:,2]=locations[:,0]+stride*deltas[:,2]
    output_boxes[:,3]=locations[:,1]+stride*deltas[:,3]


    return output_boxes


def fcos_make_centerness_targets(deltas: torch.Tensor):
    """
    Given LTRB deltas of GT boxes, compute GT targets for supervising the
    centerness regression predictor. See `fcos_get_deltas_from_locations` on
    how deltas are computed. If GT boxes are "background" => deltas are
    `(-1, -1, -1, -1)`, then centerness should be `-1`.

    For reference, centerness equation is available in FCOS paper
    https://arxiv.org/abs/1904.01355 (Equation 3).

    Args:
        deltas: Tensor of shape `(N, 4)` giving LTRB deltas for GT boxes.

    Returns:
        torch.Tensor
            Tensor of shape `(N, )` giving centerness regression targets.
    """
    centerness = None
    left, top, right, bottom = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]

    centerness = torch.sqrt((torch.min(left, right) / torch.max(left, right)) * 
                            (torch.min(top, bottom) / torch.max(top, bottom)))

    # Handling the case where GT boxes are "background" with deltas (-1, -1, -1, -1)
    # In this case, setting centerness to -1
    centerness[(left == -1) | (top == -1) | (right == -1) | (bottom == -1)] = -1


    return centerness


class FCOS(nn.Module):
    """
    FCOS: Fully-Convolutional One-Stage Detector

    This class puts together everything you implemented so far. It contains a
    backbone with FPN, and prediction layers (head). It computes loss during
    training and predicts boxes during inference.
    """

    def __init__(
        self, num_classes: int, fpn_channels: int, stem_channels: List[int]
    ):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = None
        self.pred_net = None
        self.backbone = DetectorBackboneWithFPN(fpn_channels)
        self.pred_net = FCOSPredictionNetwork(num_classes,fpn_channels,stem_channels)

        self._normalizer = 150  # per image

    def forward(
        self,
        images: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
        test_score_thresh: Optional[float] = None,
        test_nms_thresh: Optional[float] = None,
    ):
        """
        Args:
            images: Batch of images, tensors of shape `(B, C, H, W)`.
            gt_boxes: Batch of training boxes, tensors of shape `(B, N, 5)`.
                `gt_boxes[i, j] = (x1, y1, x2, y2, C)` gives information about
                the `j`th object in `images[i]`. The position of the top-left
                corner of the box is `(x1, y1)` and the position of bottom-right
                corner of the box is `(x2, x2)`. These coordinates are
                real-valued in `[H, W]`. `C` is an integer giving the category
                label for this bounding box. Not provided during inference.
            test_score_thresh: During inference, discard predictions with a
                confidence score less than this value. Ignored during training.
            test_nms_thresh: IoU threshold for NMS during inference. Ignored
                during training.

        Returns:
            Losses during training and predictions during inference.
        """

        ######################################################################
        # to obtain model predictions at every FPN location.                 #
        # Get dictionaries of keys {"p3", "p4", "p5"} giving predicted class #
        # logits, deltas, and centerness.                                    #
        ######################################################################
        # Feel free to delete this line: (but keep variable names same)
        pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = None, None, None
        # Replace "pass" statement with your code
        fpn_info = self.backbone(images)
        pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = self.pred_net(fpn_info)


        locations_per_fpn_level = None
        # Replace "pass" statement with your code
        fpn_shape = {"p3":fpn_info["p3"].shape, "p4":fpn_info["p4"].shape, "p5":fpn_info["p5"].shape}
        locations_per_fpn_level = get_fpn_location_coords(fpn_shape,self.backbone.fpn_strides, device=images.device)


        if not self.training:
            # During inference, just go to this method and skip rest of the
            # forward pass.
            # fmt: off
            return self.inference(
                images, locations_per_fpn_level,
                pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits,
                test_score_thresh=test_score_thresh,
                test_nms_thresh=test_nms_thresh,
            )
            # fmt: on


        matched_gt_boxes = []
        # Replace "pass" statement with your code

        for i in range(images.shape[0]):
            matched_gt_boxes.append(fcos_match_locations_to_gt(locations_per_fpn_level, self.backbone.fpn_strides, gt_boxes[i,:,:]))

        # Calculate GT deltas for these matched boxes. Similar structure
        # as `matched_gt_boxes` above. Fill this list:
        matched_gt_deltas = []

        # Replace "pass" statement with your code

        for i in range(images.shape[0]):
            matched_delta={}
            for level_name,feat_location in locations_per_fpn_level.items():
                matched_delta[level_name] = fcos_get_deltas_from_locations(feat_location, matched_gt_boxes[i][level_name], self.backbone.fpn_strides[level_name])
            matched_gt_deltas.append(matched_delta)

        # Calculate predicted boxes from the predicted deltas. Similar structure
        # as `matched_gt_boxes` above. Fill this list:
        pred_boxes=[]
        # Replace "pass" statement with your code

        for i in range(images.shape[0]):
            pred_box={}
            for level_name in locations_per_fpn_level.keys():
                pred_box[level_name] = fcos_apply_deltas_to_locations(pred_boxreg_deltas[level_name][i], locations_per_fpn_level[level_name], self.backbone.fpn_strides[level_name])
            pred_boxes.append(pred_box)


        matched_gt_boxes = default_collate(matched_gt_boxes)
        matched_gt_deltas = default_collate(matched_gt_deltas)
        pred_boxes= default_collate(pred_boxes)


        matched_gt_boxes = self._cat_across_fpn_levels(matched_gt_boxes)
        matched_gt_deltas = self._cat_across_fpn_levels(matched_gt_deltas)
        pred_cls_logits = self._cat_across_fpn_levels(pred_cls_logits)
        pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)
        pred_ctr_logits = self._cat_across_fpn_levels(pred_ctr_logits)
        pred_boxes = self._cat_across_fpn_levels(pred_boxes)

        # Perform EMA update of normalizer by number of positive locations.
        num_pos_locations = (matched_gt_boxes[:, :, 4] != -1).sum()
        pos_loc_per_image = num_pos_locations.item() / images.shape[0]
        self._normalizer = 0.9 * self._normalizer + 0.1 * pos_loc_per_image


        loss_cls, loss_box, loss_ctr = None, None, None

        gt_classes = matched_gt_boxes[:,:,4].clone()
        bg_mask = gt_classes==-1
        gt_classes[bg_mask] = 0
        gt_classes_one_hot = torch.nn.functional.one_hot(gt_classes.long(),self.num_classes)
        gt_classes_one_hot = gt_classes_one_hot.to(gt_boxes.dtype)
        gt_classes_one_hot[bg_mask] = 0
        loss_cls = sigmoid_focal_loss(inputs=pred_cls_logits, targets=gt_classes_one_hot)


        pred_boxreg_deltas = pred_boxreg_deltas.reshape(-1,4)
        matched_gt_deltas = matched_gt_deltas.reshape(-1,4)
        # Find the background images
        matched_boxes = matched_gt_boxes[:,:,4].clone().reshape(-1)
        background_mask = matched_boxes==-1        
        #Calculate the box loss
        loss_box = torchvision.ops.generalized_box_iou_loss(pred_boxes.reshape(-1,4), matched_gt_boxes[:,:,:4].reshape(-1,4),reduction="none")
        # Do not count the loss of background images
        loss_box[background_mask] = 0

        pred_ctr_logits = pred_ctr_logits.view(-1)
        gt_centerness = fcos_make_centerness_targets(matched_gt_deltas)
        loss_ctr = F.binary_cross_entropy_with_logits(pred_ctr_logits, gt_centerness, reduction="none")
        loss_ctr[gt_centerness<=0] = 0

        # Sum all locations and average by the EMA of foreground locations.
        # In training code, we simply add these three and call `.backward()`
        return {
            "loss_cls": loss_cls.sum() / (self._normalizer * images.shape[0]),
            "loss_box": loss_box.sum() / (self._normalizer * images.shape[0]),
            "loss_ctr": loss_ctr.sum() / (self._normalizer * images.shape[0]),
        }

    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        """
        Convert a dict of tensors across FPN levels {"p3", "p4", "p5"} to a
        single tensor. Values could be anything - batches of image features,
        GT targets, etc.
        """
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)



    def inference(
        self,
        images: torch.Tensor,
        locations_per_fpn_level: Dict[str, torch.Tensor],
        pred_cls_logits: Dict[str, torch.Tensor],
        pred_boxreg_deltas: Dict[str, torch.Tensor],
        pred_ctr_logits: Dict[str, torch.Tensor],
        test_score_thresh: float = 0.3,
        test_nms_thresh: float = 0.5,
    ):
        """
        Run inference on a single input image (batch size = 1). Other input
        arguments are same as those computed in `forward` method. This method
        should not be called from anywhere except from inside `forward`.

        Returns:
            Three tensors:
                - pred_boxes: Tensor of shape `(N, 4)` giving *absolute* XYXY
                  co-ordinates of predicted boxes.

                - pred_classes: Tensor of shape `(N, )` giving predicted class
                  labels for these boxes (one of `num_classes` labels). Make
                  sure there are no background predictions (-1).

                - pred_scores: Tensor of shape `(N, )` giving confidence scores
                  for predictions: these values are `sqrt(class_prob * ctrness)`
                  where class_prob and ctrness are obtained by applying sigmoid
                  to corresponding logits.
        """

        # Gather scores and boxes from all FPN levels in this list. Once
        # gathered, we will perform NMS to filter highly overlapping predictions.
        pred_boxes_all_levels = []
        pred_classes_all_levels = []
        pred_scores_all_levels = []

        for level_name in locations_per_fpn_level.keys():

            # Get locations and predictions from a single level.
            # We index predictions by `[0]` to remove batch dimension.
            level_locations = locations_per_fpn_level[level_name]
            level_cls_logits = pred_cls_logits[level_name][0]
            level_deltas = pred_boxreg_deltas[level_name][0]
            level_ctr_logits = pred_ctr_logits[level_name][0]

            level_pred_boxes, level_pred_classes, level_pred_scores = (
                None,
                None,
                None,  # Need tensors of shape: (N, 4) (N, ) (N, )
            )

            # Compute geometric mean of class logits and centerness:
            level_pred_scores = torch.sqrt(
                level_cls_logits.sigmoid_() * level_ctr_logits.sigmoid_()
            )

            level_pred_scores, predClasses = level_pred_scores.max(dim=1)


            retain = level_pred_scores > test_score_thresh
            level_pred_classes = predClasses[retain]
            level_pred_scores = level_pred_scores[retain]

            level_pred_boxes = fcos_apply_deltas_to_locations(level_deltas, level_locations,stride=self.backbone.fpn_strides[level_name])
            level_pred_boxes = level_pred_boxes[retain]
            removeBackground = (level_deltas[retain].sum(dim=1) != -4)
            level_pred_scores = level_pred_scores[removeBackground]
            level_pred_classes = level_pred_classes[removeBackground]
            level_pred_boxes = level_pred_boxes[removeBackground]


            dim2,dim3 = images.shape[2], images.shape[3]

            level_pred_boxes[:,0] = level_pred_boxes[:,0].clip(min=0)
            level_pred_boxes[:,1] = level_pred_boxes[:,1].clip(min=0)
            level_pred_boxes[:,2] = level_pred_boxes[:,2].clip(max=dim2)
            level_pred_boxes[:,3] = level_pred_boxes[:,3].clip(max=dim3)



            pred_boxes_all_levels.append(level_pred_boxes)
            pred_classes_all_levels.append(level_pred_classes)
            pred_scores_all_levels.append(level_pred_scores)

        # Combine predictions from all levels and perform NMS.
        pred_boxes_all_levels = torch.cat(pred_boxes_all_levels)
        pred_classes_all_levels = torch.cat(pred_classes_all_levels)
        pred_scores_all_levels = torch.cat(pred_scores_all_levels)

        keep = class_spec_nms(
            pred_boxes_all_levels,
            pred_scores_all_levels,
            pred_classes_all_levels,
            iou_threshold=test_nms_thresh,
        )
        pred_boxes_all_levels = pred_boxes_all_levels[keep]
        pred_classes_all_levels = pred_classes_all_levels[keep]
        pred_scores_all_levels = pred_scores_all_levels[keep]
        return (
            pred_boxes_all_levels,
            pred_classes_all_levels,
            pred_scores_all_levels,
        )
