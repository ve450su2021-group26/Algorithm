import torch


def normalization_by_range(t):
    max = torch.max(t)
    min = torch.min(t)
    normalized_t = (t - min) / (max - min)

    return normalized_t


from einops import rearrange


def tensor_to_np(img_tensor):
    img = img_tensor.mul(255).byte()
    img = img.numpy()
    img = rearrange(img, '(c h w -> c w h)')
    return img
