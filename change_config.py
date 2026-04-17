#!/usr/bin/env python3
"""
Generate an RPN-focused override config from a base Faster R-CNN config,
while KEEPING the original base test pipeline.

Example:
python make_rpn_config_keep_pipeline.py \
    --base-config /path/to/fg_frcnn_dota_pretrain_sar_wavelet_r50.py \
    --out-config /path/to/generated_rpn_fg_frcnn_dota_pretrain_sar_wavelet_r50.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent


def build_override_text(
    base_config: str,
    rpn_nms_pre: int,
    rpn_max_per_img: int,
    rpn_nms_iou: float,
    rcnn_score_thr: float,
    rcnn_nms_iou: float,
    rcnn_max_per_img: int,
    cache_max_proposals: int,
    cache_score_threshold: float,
    cache_nms_threshold: float,
    save_format: str,
) -> str:
    return dedent(
        f"""
        # Auto-generated RPN proposal config
        # Base config:
        _base_ = {base_config!r}

        # Keep many proposals from the RPN for proposal mining / few-shot learning.
        model = dict(
            test_cfg=dict(
                rpn=dict(
                    nms_pre={rpn_nms_pre},
                    max_per_img={rpn_max_per_img},
                    nms=dict(type='nms', iou_threshold={rpn_nms_iou}),
                    min_bbox_size=0,
                ),
                # These RCNN settings matter less if you only extract RPN proposals,
                # but keeping them loose can help with debugging/comparison.
                rcnn=dict(
                    score_thr={rcnn_score_thr},
                    nms=dict(type='nms', iou_threshold={rcnn_nms_iou}),
                    max_per_img={rcnn_max_per_img},
                ),
            )
        )

        # Optional cache config used by custom proposal-extraction scripts.
        rpn_cache_cfg = dict(
            max_proposals_per_image={cache_max_proposals},
            score_threshold={cache_score_threshold},
            nms_threshold={cache_nms_threshold},
            save_format={save_format!r},
        )

        # NOTE:
        # We intentionally do NOT override test_pipeline here.
        # The base config's original test pipeline will be used as-is.
        """
    ).strip() + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True, help="Path to base FRCNN config")
    parser.add_argument("--out-config", required=True, help="Path to output generated config")
    parser.add_argument("--rpn-nms-pre", type=int, default=12000)
    parser.add_argument("--rpn-max-per-img", type=int, default=4000)
    parser.add_argument("--rpn-nms-iou", type=float, default=0.9)
    parser.add_argument("--rcnn-score-thr", type=float, default=0.001)
    parser.add_argument("--rcnn-nms-iou", type=float, default=0.7)
    parser.add_argument("--rcnn-max-per-img", type=int, default=1000)
    parser.add_argument("--cache-max-proposals", type=int, default=4000)
    parser.add_argument("--cache-score-threshold", type=float, default=0.0)
    parser.add_argument("--cache-nms-threshold", type=float, default=0.9)
    parser.add_argument("--save-format", choices=["pickle", "npz", "json"], default="pickle")
    args = parser.parse_args()

    out_path = Path(args.out_config)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = build_override_text(
        base_config=args.base_config,
        rpn_nms_pre=args.rpn_nms_pre,
        rpn_max_per_img=args.rpn_max_per_img,
        rpn_nms_iou=args.rpn_nms_iou,
        rcnn_score_thr=args.rcnn_score_thr,
        rcnn_nms_iou=args.rcnn_nms_iou,
        rcnn_max_per_img=args.rcnn_max_per_img,
        cache_max_proposals=args.cache_max_proposals,
        cache_score_threshold=args.cache_score_threshold,
        cache_nms_threshold=args.cache_nms_threshold,
        save_format=args.save_format,
    )

    out_path.write_text(text, encoding="utf-8")
    print(f"Saved generated config to: {out_path}")


if __name__ == "__main__":
    main()