import timm
import torch

from einops import rearrange
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class imageInferenceModel():
    def __init__(self, resize_size, weight_path, device):
        self.transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        self.net = timm.create_model('efficientnet_b0', num_classes=9)
        self.net.to(device)
        self.net.load_state_dict(torch.load(weight_path))
        self.device = device

    def predict(self, image):
        # image is a PIL image
        with torch.no_grad():
            img = self.transform(image).to(self.device)
            img = rearrange(img, 'c h w -> () c h w')
            label = torch.argmax(self.net(img))
        return label.item()
