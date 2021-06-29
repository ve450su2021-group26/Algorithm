from u2net import u2net, u2netp
import torch
from pathlib2 import Path

NETS = {
    'u2net': (u2net, Path('./weights/u2net/u2net.pth')),
    'u2netp': (u2netp, Path('./weights/u2net/u2netp.pth'))
}


class SaliencyObjDetect:
    def __init__(self, net_type='u2net'):
        assert torch.cuda.is_available(
        ), 'This module is only avaliable in cuda device.'

        # initial net
        net_class, net_weight_path = NETS[net_type][0]
        self.net = net_class()
        self.net.load_state_dict(torch.load(str(net_weight_path)))
        self.net.cuda()
        self.net.eval()

    def detect_saliency(self):
        pass
