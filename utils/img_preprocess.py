import torch
import torchvision

from torchvision import transform


class Rescale:
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        # img = transform.resize(image,(new_h,new_w),mode='constant')
        # lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

        img = transform.resize(image, (self.output_size, self.output_size),
                               mode='constant')
        lbl = transform.resize(label, (self.output_size, self.output_size),
                               mode='constant',
                               order=0,
                               preserve_range=True)

        return {'imidx': imidx, 'image': img, 'label': lbl}


class ImgPreprocessor:
    def __init__(self):
        pass
