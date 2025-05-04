from monai.networks.nets import UNet
from config.default_config import Config

def create_model():
    """Create 3D UNet model with improved architecture for binary segmentation"""
    model = UNet(
        spatial_dims=3,
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        channels=Config.CHANNELS,
        strides=Config.STRIDES,
        num_res_units=Config.NUM_RES_UNITS,
        dropout=Config.DROPOUT,
        norm="instance",
        act="leakyrelu"
    ).to(Config.DEVICE)
    return model