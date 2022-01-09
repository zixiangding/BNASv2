# BNASv2
Authors' implementation of "BNAS-v2: Memory-efficient and Performance-collapse-prevented Broad Neural Architecture Search" (2022) in Pytorch.
Includes code for CIFAR-10 and ImageNet image classification.

Paper:  https://arxiv.org/pdf/2009.08886.pdf

Authors: Zixiang Ding, Yaran Chen, Nannan Li and Dongbin Zhao. 
## Requirements
```
BNASv2CLR: Python >= 3.5.5, PyTorch == 0.3.1, torchvision == 0.2.0
BNASv2PC: Python >= 3.5.5, PyTorch == 0.4.0, torchvision == 0.2.0
```
NOTE: PyTorch 0.4 is not supported for BNASv2CLR and would lead to OOM.

## Datasets
CIFAR-10 can be automatically downloaded by torchvision, ImageNet needs to be manually downloaded (preferably to a SSD) following the instructions [here](https://github.com/pytorch/examples/tree/master/imagenet).

## Architecture search on CIFAR-10
To carry out architecture search using BNASv2CLR run
```
cd BNASv2CLR && python train_search_b_confidence.py             # for cells on CIFAR-10 using 1nd-order approximation
cd BNASv2CLR && python train_search_b_confidence.py --unrolled  # for cells on CIFAR-10 using 2nd-order approximation
```
To carry out architecture search using BNASv2PC, run
```
cd BNASv2PC && python train_search_b.py             # for cells on CIFAR-10 using 1nd-order approximation
cd BNASv2PC && python train_search_b.py --unrolled  # for cells on CIFAR-10 using 2nd-order approximation
```

## Architecture evaluation on CIFAR-10
To evaluate our best cells of BNASv2CLR by training from scratch, run
```
cd BNASv2CLR && python train_b.py
```
To evaluate our best cells of BNASv2PC by training from scratch, run
```
cd BNASv2PC && python train_b.py
```
Customized architectures are supported through the `--arch` flag once specified in `genotypes.py`.

## Architecture search on ImageNet

To carry out architecture search on ImageNet using B-PC-DARTS, run
```
cd BNASv2PC && python imagenet_data_sample.py     # for sampleing subset of ImageNet for proxyless search
python train_search_imagenet_b.py                   # for architecture search on ImageNet
```

## Architecture evaluation on ImageNet
To evaluate our best cells for ImageNet by training from scratch, run
```
cd BNASv2PC && python train_imagenet_b.py
```

## Citation
If you use any part of this code in your research, please cite our [paper](https://arxiv.org/pdf/2009.08886.pdf):
```
@article{ding2021bnas,
title={BNAS-v2: Memory-efficient and Performance-collapse-prevented Broad Neural Architecture Search},
author={Ding, Zixiang and Chen, Yaran and Li, Nannan and Zhao, Dongbin},
journal={arXiv preprint arXiv:2009.08886},
year={2021}
}
```
