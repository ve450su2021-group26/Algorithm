import torch


class SaliencyObjDetect:
    def __init__(self, model_type='u2net'):
        assert torch.cuda.is_available(
        ), 'This module is only avaliable in cuda device.'

    def detect_saliency(self):
        pass
