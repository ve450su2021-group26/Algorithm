# Usage
## Image Module
```python
from img_module.inference_model import imageInferenceModel

# the device for you want to run the model
device = 'cuda'
model = imageInferenceModel(156, device=device)
# img is PIL Image format
model.predict(img)
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
