dependencies = ['torch']
from mono.utils.logger import setup_logger
from mono.utils.comm import init_env
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.utils.do_test import do_scalecano_test_with_custom_data
from mono.utils.mldb import load_data_info, reset_ckpt_path
from mono.utils.custom_data import load_from_annos, load_data
import torch

def Metric3D(pretrained=False, **kwargs):
    """
    # This docstring shows up in hub.help()
    MonoDepth model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # monodepth model
    model = get_configured_monodepth_model(**kwargs)
    if pretrained:
        ckpt = torch.hub.load_state_dict_from_url(
            'https://path/to/pretrained/weights',
            progress=True,
            check_hash=True
        )
        model.load_state_dict(ckpt)
    return model