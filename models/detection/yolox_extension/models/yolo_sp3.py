"""
Original Yolox PAFPN code with slight modifications
"""
from typing import Dict, Optional, Tuple

import torch as th
import torch.nn as nn
import numpy as np

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from ...yolox.models.network_blocks import BaseConv, CSPLayer, DWConv
from ...utils import window_reverse, window_partition, MLP, MixerBlock
from data.utils.types import BackboneFeatures


class YOLOSP3(nn.Module):
    """
    Removed the direct dependency on the backbone.
    """

    def __init__(
            self,
            in_stages: Tuple[int, ...] = (2, 3, 4),
            in_channels: Tuple[int, ...] = (256, 512, 1024),
            patch_dim: int = 8,
            feat_channels: Tuple[int, ...] = (8, 16, 128),
            mixer_count: int = 1,
            compile_cfg: Optional[Dict] = None,
            **kwargs,
    ):
        super().__init__()
        self.feat_channels = feat_channels
        assert len(in_stages) == len(in_channels)
        assert len(in_channels) == 3, 'Current implementation only for 3 feature maps'
        self.in_features = in_stages
        self.in_channels = in_channels
        self.backlinks = []

        ###### Compile if requested ######
        if compile_cfg is not None:
            compile_mdl = compile_cfg['enable']
            if compile_mdl and th_compile is not None:
                self.forward = th_compile(self.forward, **compile_cfg['args'])
            elif compile_mdl:
                print('Could not compile PAFPN because torch.compile is not available')

        ##################################

        self.num_ins = len(in_channels)
        self.mixer_count = mixer_count
        self.patch_dim = patch_dim
        self.feat_channels = feat_channels
        self.mix_nodes = [64, 64, 64]
        self.start_idx = 0

        self.last_featmap_dim = self.patch_dim // 8
        self.out_lvl = 2
        self.out_patch_dim = self._lvl_patch_dim(2)

        self.intpr = nn.ModuleList()
        self.inner = nn.ModuleList()
        self.final_norm_act = nn.ModuleList()

        for i in range(len(self.feat_channels)):
            tokens = self._feat_tokens(i)
            exposure = tokens * self.feat_channels[i]
            self.intpr.append(
                nn.Sequential(
                    nn.Linear(self.in_channels[i] * tokens, exposure),
                    nn.LayerNorm(exposure)
                )
                if i >= self.start_idx else None
            )
            if i == 0:
                self.inner.append(nn.Identity())
                self.final_norm_act.append(nn.Sequential(nn.GELU()))
            else:
                self.inner.append(nn.Linear(exposure + self.mix_nodes[i - 1], self.mix_nodes[i]))
                self.final_norm_act.append(nn.Sequential(nn.LayerNorm(self.mix_nodes[i]), nn.GELU()))

        self.mixers = nn.ModuleList()
        for i in range(len(self.feat_channels)):
            if self.mixer_count > 0:
                self.mixers.append(nn.Sequential(*[
                    MixerBlock((self._lvl_patch_dim(i)) ** 2, self.mix_nodes[i]) for _ in range(self.mixer_count)
                ]))

        ###### Compile if requested ######
        if compile_cfg is not None:
            compile_mdl = compile_cfg['enable']
            if compile_mdl and th_compile is not None:
                self.forward = th_compile(self.forward, **compile_cfg['args'])
            elif compile_mdl:
                print('Could not compile PAFPN because torch.compile is not available')
        ##################################

    def _feat_tokens(self, lvl):
        return self._lvl_patch_dim(lvl) ** 2

    def _lvl_patch_dim(self, lvl):
        _min = self.last_featmap_dim
        lvl = 2 - lvl
        return _min * 2 ** (lvl)

    def forward(self, input: BackboneFeatures):
        """
        Args:
            inputs: Feature maps from backbone

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        debug = False

        inputs = [input[f] for f in self.in_features]
        x2, x1, x0 = inputs

        B, _, H4, W4 = x2.shape
        self.backlinks[0].yolox_head.width = H4
        self.backlinks[0].yolox_head.height = W4

        parts = []
        feats = None
        for i in range(self.start_idx, len(self.feat_channels)):

            part = window_partition(inputs[i], self._lvl_patch_dim(i), channel_last=False)
            print("Shape after partition: ", part.shape) if debug else None
            part = th.flatten(part, -2)
            print("Shape after flatten: ", part.shape) if debug else None
            part = self.intpr[i](part)
            print("Shape after reduction: ", part.shape) if debug else None
            if feats is None:
                feats = part
            else:
                feats = th.cat([feats, part], dim=-1)
            feats = self.inner[i](feats)
            parts.append(feats)

        B, T, _ = parts[-1].shape

        outputs = []
        for i, part in enumerate(parts):

            out = part.view(B, T, self._lvl_patch_dim(i) ** 2, self.out_channels)
            out = self.final_norm_act[i](out)
            print("Shape after view: ", out.shape) if debug else None

            if self.mixers is not None:
                outputs.append(self.mixers[i](out))

        ######## Only for debugging ########
        for i, output in enumerate(outputs):
            ds_factor = 2 ** i
            feats = window_reverse(output, self._lvl_patch_dim(i), H4 // ds_factor, W4 // ds_factor)
            print("Shape after windows reverse: ", feats.shape) if debug else None
        exit() if debug else None

        return outputs
