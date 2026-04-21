CUDNN_BENCHMARK: false

DATASETS:
  TRAIN: ("sar_agnostic_train",)
  TEST: ("sar_agnostic_val",)
  PROPOSAL_FILES_TRAIN: []
  PROPOSAL_FILES_TEST: []
  PRECOMPUTED_PROPOSAL_TOPK_TRAIN: 2000
  PRECOMPUTED_PROPOSAL_TOPK_TEST: 1000

DATALOADER:
  NUM_WORKERS: 4
  FILTER_EMPTY_ANNOTATIONS: true
  SAMPLER_TRAIN: "TrainingSampler"
  ASPECT_RATIO_GROUPING: true

INPUT:
  FORMAT: "BGR"
  MASK_FORMAT: "polygon"
  MIN_SIZE_TRAIN: (640, 800, 960)
  MAX_SIZE_TRAIN: 1600
  MIN_SIZE_TRAIN_SAMPLING: "choice"
  MIN_SIZE_TEST: 800
  MAX_SIZE_TEST: 1600
  RANDOM_FLIP: "horizontal"
  CROP:
    ENABLED: false

MODEL:
  META_ARCHITECTURE: "GeneralizedRCNN"
  DEVICE: "cuda"
  WEIGHTS: "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
  PIXEL_MEAN: [128.0, 128.0, 128.0]
  PIXEL_STD: [64.0, 64.0, 64.0]

  BACKBONE:
    NAME: "build_resnet_fpn_backbone"
    FREEZE_AT: 2

  RESNETS:
    DEPTH: 50
    OUT_FEATURES: ["res3", "res4", "res5"]
    NUM_GROUPS: 1
    NORM: "FrozenBN"
    WIDTH_PER_GROUP: 64
    STEM_OUT_CHANNELS: 64
    RES2_OUT_CHANNELS: 256
    STRIDE_IN_1X1: true
    RES5_DILATION: 1
    DEFORM_ON_PER_STAGE: [false, false, false, false]
    DEFORM_MODULATED: false
    DEFORM_NUM_GROUPS: 1

  FPN:
    IN_FEATURES: ["res3", "res4", "res5"]
    OUT_CHANNELS: 256
    NORM: ""
    FUSE_TYPE: "sum"

  ANCHOR_GENERATOR:
    NAME: "DefaultAnchorGenerator"
    SIZES: [[16], [32], [64], [128], [256]]
    ASPECT_RATIOS: [[0.5, 1.0, 2.0]]
    ANGLES: [[-90, 0, 90]]
    OFFSET: 0.0

  PROPOSAL_GENERATOR:
    NAME: "RPN"
    MIN_SIZE: 0

  RPN:
    HEAD_NAME: "StandardRPNHead"
    IN_FEATURES: ["p3", "p4", "p5", "p6", "p7"]
    BOUNDARY_THRESH: -1
    IOU_THRESHOLDS: [0.3, 0.7]
    IOU_LABELS: [0, -1, 1]
    BATCH_SIZE_PER_IMAGE: 256
    POSITIVE_FRACTION: 0.5
    PRE_NMS_TOPK_TRAIN: 2000
    PRE_NMS_TOPK_TEST: 2000
    POST_NMS_TOPK_TRAIN: 2000
    POST_NMS_TOPK_TEST: 1500
    NMS_THRESH: 0.8
    LOSS_WEIGHT: 1.0
    BBOX_REG_LOSS_TYPE: "smooth_l1"
    BBOX_REG_LOSS_WEIGHT: 1.0
    BBOX_REG_WEIGHTS: [1.0, 1.0, 1.0, 1.0]
    SMOOTH_L1_BETA: 0.0

  ROI_HEADS:
    NAME: "StandardROIHeads"
    IN_FEATURES: ["p3", "p4", "p5", "p6", "p7"]
    NUM_CLASSES: 1
    BATCH_SIZE_PER_IMAGE: 512
    POSITIVE_FRACTION: 0.25
    PROPOSAL_APPEND_GT: true
    IOU_THRESHOLDS: [0.5]
    IOU_LABELS: [0, 1]
    SCORE_THRESH_TEST: 0.001
    NMS_THRESH_TEST: 0.5

  ROI_BOX_HEAD:
    NAME: "FastRCNNConvFCHead"
    NUM_CONV: 0
    NUM_FC: 2
    FC_DIM: 1024
    CONV_DIM: 256
    NORM: ""
    POOLER_RESOLUTION: 7
    POOLER_SAMPLING_RATIO: 0
    POOLER_TYPE: "ROIAlignV2"
    CLS_AGNOSTIC_BBOX_REG: true
    TRAIN_ON_PRED_BOXES: false
    SMOOTH_L1_BETA: 0.0
    BBOX_REG_LOSS_TYPE: "smooth_l1"
    BBOX_REG_LOSS_WEIGHT: 1.0
    BBOX_REG_WEIGHTS: [10.0, 10.0, 5.0, 5.0]
    USE_FED_LOSS: false
    USE_SIGMOID_CE: false

SOLVER:
  AMP:
    ENABLED: false
  IMS_PER_BATCH: 8
  BASE_LR: 0.0025
  MAX_ITER: 20000
  STEPS: (12000, 16000)
  GAMMA: 0.1
  WARMUP_ITERS: 1000
  WARMUP_FACTOR: 0.001
  WARMUP_METHOD: "linear"
  MOMENTUM: 0.9
  WEIGHT_DECAY: 0.0001
  WEIGHT_DECAY_NORM: 0.0
  CHECKPOINT_PERIOD: 2000
  CLIP_GRADIENTS:
    ENABLED: true
    CLIP_TYPE: "norm"
    CLIP_VALUE: 35.0
    NORM_TYPE: 2.0

TEST:
  EVAL_PERIOD: 2000
  DETECTIONS_PER_IMAGE: 1000
  AUG:
    ENABLED: false

OUTPUT_DIR: "./output/sar_frcnn_r50_fpn_objectness"
VERSION: 2
VIS_PERIOD: 0
SEED: 42


import os
from detectron2.engine import DefaultTrainer, default_argument_parser, launch
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo

def setup():
    cfg = get_cfg()
    cfg.merge_from_file("configs/sar_frcnn_r50_fpn_objectness.yaml")

    register_coco_instances(
        "sar_agnostic_train",
        {},
        "/path/to/data/sar_agnostic/train_coco.json",
        "/path/to/data/sar_agnostic/train",
    )
    register_coco_instances(
        "sar_agnostic_val",
        {},
        "/path/to/data/sar_agnostic/val_coco.json",
        "/path/to/data/sar_agnostic/val",
    )

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

def main():
    cfg = setup()
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main()