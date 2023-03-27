# CM-TStega

* This repository is our implementation of CM-TStega.
* Paper: Cross-modal Text Steganography Against Synonym Substitution-Based Text attack ([2023 IEEE SPL](https://ieeexplore.ieee.org/abstract/document/10075392))
* Authors: Wanli Peng, Tao Wang, Zhenxing Qian, Sheng Li, Xinpeng Zhang
#### The backbone of the CM-TStega is a popular Image Captioning Model: [AoANet](https://arxiv.org/abs/1908.06954) (the code is [here!](https://github.com/husthuaan/AoANet))

# Requirements
* Python == 3.8
* torch == 1.13
* torchtext == 0.14.0
* torchvision == 0.14.0
* tensorboardX == 2.5.1
* loguru == 0.6.0
* six == 1.16.0

#### The example code of training stage of CM-TStega (train_2.sh)
#### The example code of inference stage of CM-TStega (test-best.sh)

# Cite
#### If this paper and code are helpful, please cite the following paper:

@ARTICLE{10075392,
  author={Peng, Wanli and Wang, Tao and Qian, Zhenxing and Li, Sheng and Zhang, Xinpeng},
  journal={IEEE Signal Processing Letters}, 
  title={Cross-Modal Text Steganography Against Synonym Substitution-Based Text Attack}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/LSP.2023.3258862}}
