import torch

from .u2net import U2NET, U2NETP
from pathlib2 import Path
from torchvision import transforms
from .utils.tensor_operation import normalization_by_range

NETS = {
    'u2net': (U2NET, Path('./weights/u2net/u2net.pth')),
    'u2netp': (U2NETP, Path('./weights/u2net/u2netp.pth'))
}

IMGNET_MEAN_RGB = (0.485, 0.456, 0.406)
IMGNET_STD_RGB = (0.229, 0.224, 0.225)


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

        # initial transforms
        self.transform = transforms.Compose([
            transforms.Resize(320),
            transforms.ToTensor(),
            transforms.Normalize(IMGNET_MEAN_RGB, IMGNET_STD_RGB)
        ])

    def get_saliency_mask(self, img):
        pass

    def detect_saliency(self, img):
        with torch.no_grad():
            img_tensor = self.transform(img).cuda().unsqueeze(0)
            d1, d2, d3, d4, d5, d6, d7 = self.net(img_tensor)
            prediction = normalization_by_range(d1[:, 0, :, :])
        return prediction
