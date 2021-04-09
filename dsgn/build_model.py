from .models import *

def build_model(cfg):
    if cfg.mono:
        if getattr(cfg, "plume", False):
            model = PLUMENet(cfg)
        else:
            model = MonoNet(cfg)
    else:
        model = StereoNet(cfg)

    return model
