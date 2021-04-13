from .models import *

def build_model(cfg):
    if cfg.mono:
        model_arc = getattr(cfg, "model", "MonoNet")
        if model_arc == "PLUMENet":
            model = PLUMENet(cfg)
        elif model_arc == "DepthNet":
            model = DepthNet(cfg)
        else:
            model = MonoNet(cfg)
    else:
        model = StereoNet(cfg)

    return model
