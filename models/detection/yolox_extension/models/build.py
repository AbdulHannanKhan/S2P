from typing import Tuple

from omegaconf import OmegaConf, DictConfig

from .yolo_pafpn import YOLOPAFPN
from .yolo_sp3 import  YOLOSP3
from ...yolox.models.yolo_head import YOLOXHead
from ...yolox.models.yolo_dfdn_head import YOLODFDNHead


def build_yolox_head(head_cfg: DictConfig, in_channels: Tuple[int, ...], strides: Tuple[int, ...]):
    head_cfg_dict = OmegaConf.to_container(head_cfg, resolve=True, throw_on_missing=True)
    head_name = head_cfg_dict.pop('name')
    head_cfg_dict.pop('version', None)
    head_cfg_dict.update({"in_channels": in_channels})
    head_cfg_dict.update({"strides": strides})
    compile_cfg = head_cfg_dict.pop('compile', None)
    head_cfg_dict.update({"compile_cfg": compile_cfg})
    if head_name in {'YOLODFNF', 'YOLO_DFDN'}:
        return YOLODFDNHead(**head_cfg_dict)
    return YOLOXHead(**head_cfg_dict)


def build_yolox_fpn(fpn_cfg: DictConfig, in_channels: Tuple[int, ...]):
    fpn_cfg_dict = OmegaConf.to_container(fpn_cfg, resolve=True, throw_on_missing=True)
    fpn_name = fpn_cfg_dict.pop('name')
    fpn_cfg_dict.update({"in_channels": in_channels})
    compile_cfg = fpn_cfg_dict.pop('compile', None)
    fpn_cfg_dict.update({"compile_cfg": compile_cfg})
    if fpn_name in {'PAFPN', 'pafpn'}:
        return YOLOPAFPN(**fpn_cfg_dict)
    elif fpn_name in {'SP3', 'sp3'}:
        fpn_cfg_dict.pop('depth', None)
        return YOLOSP3(**fpn_cfg_dict)
    raise NotImplementedError
