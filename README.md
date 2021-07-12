# Usage
## Image Module
* predcition
```python
from img_module.inference_model import ImageInferenceModel
from PIL import Image
# the device which the model will run on it
device = 'cuda'
model = ImageInferenceModel(224, device=device)
# img is PIL Image format
model.predict(img)
```
* saliency object detection
```python
from img_module.saliency import SaliencyObjDetecter
from PIL import Image
from img_module.utils.img_preprocess import tensor_to_cv2
# the device which the model will run on it
device = 'cuda'
detecter = SaliencyObjDetecter(device=device)

front,back = detecter.get_front_back_ground(img)

# the frontground and background are cv2 format now
front = tensor_to_cv2(front)
back = tensor_to_cv2(back)
```
# Reference
## [U^2Net](https://github.com/xuebinqin/U-2-Net/blob/master/README.md)
[original implementation we used here](https://github.com/xuebinqin/U-2-Net)
```
@InProceedings{Qin_2020_PR,
title = {U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection},
author = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood and Zaiane, Osmar and Jagersand, Martin},
journal = {Pattern Recognition},
volume = {106},
pages = {107404},
year = {2020}
}
```
## timm
```
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```
## [Meta Pseudo Labels](https://arxiv.org/abs/2003.10580)
[unofficial implementation by kekmodel we used here](https://github.com/kekmodel/MPL-pytorch)
```
@misc{assran2021semisupervised,
      title={Semi-Supervised Learning of Visual Features by Non-Parametrically Predicting View Assignments with Support Samples}, 
      author={Mahmoud Assran and Mathilde Caron and Ishan Misra and Piotr Bojanowski and Armand Joulin and Nicolas Ballas and Michael Rabbat},
      year={2021},
      eprint={2104.13963},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
