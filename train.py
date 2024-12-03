#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# training executable for MASt3R
# --------------------------------------------------------
from mast3r.model import AsymmetricMASt3R
from mast3r.losses import ConfMatchingLoss, MatchingLoss, APLoss, Regr3D, InfoNCE, Regr3D_ScaleShiftInv
from mast3r.datasets import ARKitScenes, BlendedMVS, Co3d, MegaDepth, ScanNetpp, StaticThings3D, Waymo, WildRGBD

import mast3r.utils.path_to_dust3r  # noqa
# add mast3r classes to dust3r imports
import dust3r.training
dust3r.training.AsymmetricMASt3R = AsymmetricMASt3R
dust3r.training.Regr3D = Regr3D
dust3r.training.Regr3D_ScaleShiftInv = Regr3D_ScaleShiftInv
dust3r.training.MatchingLoss = MatchingLoss
dust3r.training.ConfMatchingLoss = ConfMatchingLoss
dust3r.training.InfoNCE = InfoNCE
dust3r.training.APLoss = APLoss

import dust3r.datasets
dust3r.datasets.ARKitScenes = ARKitScenes
dust3r.datasets.BlendedMVS = BlendedMVS
dust3r.datasets.Co3d = Co3d
dust3r.datasets.MegaDepth = MegaDepth
dust3r.datasets.ScanNetpp = ScanNetpp
dust3r.datasets.StaticThings3D = StaticThings3D
dust3r.datasets.Waymo = Waymo
dust3r.datasets.WildRGBD = WildRGBD

from dust3r.training import get_args_parser as dust3r_get_args_parser  # noqa
from dust3r.training import train  # noqa


def get_args_parser():
    parser = dust3r_get_args_parser()
    # change defaults
    parser.prog = 'MASt3R training'
    parser.set_defaults(
        model="AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True)"
    )
    return parser


if __name__ == '__main__':
    from dust3r.datasets import ColorJitter
    # ds = ScanNetpp(
    #     split="train",
    #     ROOT="/home/morkru/scannet/preprocessed",
    #     resolution=[(512, 160)], # (512, 384), (512, 336), (512, 288), (512, 256), (512, 160)],
    #     aug_crop=False, aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5
    # )
    args = get_args_parser()
    args = args.parse_args()
    train(args)
