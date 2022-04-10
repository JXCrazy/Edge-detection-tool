# Edge-detection-tool
Noise-robust edge detection, including some codes of detecting edge.
Readme of Pakage of Edge-detection-tool

1. “Canny-detector.m” is the Canny detector equipped by the contrast equalization and noise-dependent lower thresholds.Refer to J. Canny, “A computational approach to edge detection,” IEEE Trans. Pattern Anal. Mach. Intell., 8(6): 679-698, 1986.M. S. Nixon and A. S. Aguado, “Feature Extraction and Image Processing,” Chapter 3, Newnes, 2002.

2. “FOM_measure.m” is the Pratt’s figure of merits of edge maps. Refer to W. K. Pratt, “Digital Image Processing,” Wiley Interscience Publications, 1978.

3.HED-UNet Combined Segmentation and Edge Detection for Monitoring the Antarctic Coastline
# HED-UNet
Code for HED-UNet, a model for simultaneous semantic segmentation and edge detection.
## Glacier Fronts
This model was originally developed to detect calving front margins in Antarctica from Sentinel-1 SAR imagery.
![glaciers](figures/glaciers.jpg)
## Building Footprints
As the original dataset isn't publicly available, this repository contains an adaption of the model for building footprint extraction on the [Inria Aerial Image Labeling dataset](https://project.inria.fr/aerialimagelabeling/). Here are some example results:
##### Bloomington
![bloomington27-overview-binary](figures/bloomington27-overview-binary.jpg)
##### Innsbruck
![innsbruck20-overview-binary](figures/innsbruck20-overview-binary.jpg)
##### San Francisco
![sfo20-overview-binary](figures/sfo20-overview-binary.jpg)
## Usage
In order to use this for your project, you will need adapt either the `get_dataloader` function in `train.py` or the methods in `data_loading.py`.
## Citation
If you find our code helpful and use it in your research, please use the following BibTeX entry.
```tex
@article{HEDUNet2021,
  author={Heidler, Konrad and Mou, Lichao and Baumhoer, Celia and Dietz, Andreas and Zhu, Xiao Xiang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={HED-UNet: Combined Segmentation and Edge Detection for Monitoring the Antarctic Coastline}, 
  year={2021},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TGRS.2021.3064606}
}
```
4.SDL-Skeleton-main
## SDL-Skeleton
**Our paper has been accepted by [TIP(2021)](https://ieeexplore.ieee.org/document/9432856), the NAS part of code and the pretrained model will come soon !**
   **SDL-Skeleton is a FREE toolbox for object skeleton detection, which also has strong adaptability to general pixel-wise binary classification tasks, such as edge detection, saliency detection, line detetection, building extraction and road extraction. This code is based on the implementation of  [HED](https://github.com/s9xie/hed) and [SRN](https://github.com/KevinKecc/SRN).**  
SDL-Skeleton includes popular skeleton detection methods, including HED[<sup>1</sup>](#hed), SRN[<sup>2</sup>](#srn), HiFi[<sup>3</sup>](#hifi), DeepFlux[<sup>4</sup>](#deepflux) and our newly proposed [Ada-LSN](https://arxiv.org/pdf/2011.03972.pdf)[<sup>5</sup>](#adalsn). Ada-LSN achieved the state-of-the-art results across all skeleton dataset, for example, we achieved **0.786** performace on sklarge dataset. 
<p align="center">
  <img src="results/result_skeleton.png" alt="skeleton" width="60%">
</p>
<p align="center">
Figure 1: Skeleton detection examples.
</p>
<p align="center">
  <img src="results/result_edge.png" alt="edge" width="60%">
</p>
<p align="center">
Figure 2: Edge detection examples.
</p>
<p align="center">
  <img src="results/result_building.png" alt="building" width="60%">
</p>
<p align="center">
Figure 3: Building extraction examples.
</p>
<p align="center">
  <img src="results/result_road.png" alt="road" width="60%">
</p>
<p align="center">
Figure 4: Road extraction examples.
</p>
## Requirements
- python 3
- pytorch >= 0.4
- torchvision
## Pretrained models
## Datasets
**Skeleton Detection**
  Five commonly used skeleton datasets are used, including [sklarge](https://kaizhao.net/sk-large)、[sk506](https://openaccess.thecvf.com/content_cvpr_2016/html/Shen_Object_Skeleton_Extraction_CVPR_2016_paper.html)、[sympascal](https://github.com/KevinKecc/SRN)、[symmax](https://link.springer.com/chapter/10.1007%2F978-3-642-33786-4_4) and [whsymmax](https://dl.acm.org/doi/10.1016/j.patcog.2015.10.015). You also can download all these datasets at [here](https://pan.baidu.com/s/1ODsK9PmUHLr15GxTN3LtBw), password:x9bd and revaluation code at [here](https://pan.baidu.com/s/1ewnpj9wpJFAOCEHA8hLABg), password:zyqn. 
  The preliminary data augmentation code can be downloaded at [sklarge](https://kaizhao.net/sk-large), including resizing images to 3 scales (0.8x, 1.0x, and 1.2x), rotating for 4 directions (0◦, 90◦, 180◦,and 270◦), flipping in 2 orientations (left-to-right and up-to-down). After that, you can use resolution normalization technology (dataRN.py), which helps for skeleton detection because of their different image size. 
 **Other tasks**
 We also test our methods on [edge detection](https://www.researchgate.net/publication/45821321_Contour_Detection_and_Hierarchical_Image_Segmentation), [building extraction](https://project.inria.fr/aerialimagelabeling/download/) and [road extraction](http://deepglobe.org/). 
## Usages
**Skeleton Detection**
Test HED and SRN by run:
```
python train.py --network 'hed'            # HED
python train.py --network 'srn'            # SRN
python train.py --network 'deep_flux'      # DeepFlux
```
At the same time, modify the saved path of the network model in engines/trainer.py. If you want to test DeepFlux, you also need to modify the data loader to use datasets/sklarge_flux.py. As for HiFi, we only implemented the network structure, with lacking the multi-scale annotation datasets.
Test Ada-LSN by run:
```
python train_AdaLSN.py
```
Our Ada-LSN supports different backbones, including VGG, ResNet, Res2Net and Inception. Simply modify the Ada_LSN/model.py to switch between different backbones. The performance of these different backbones on the sklarge dataset is as follows：
|backbones |  VGG  | ResNet50 | Res2Net | InceptionV3 |
|  ----    | ----  | -------- | ------- | ----------- |
| F-score  | 0.763 |  0.764   |  0.768  |  0.786      |
 **Other tasks** 
 Our Ada-LSN also can be used for other pixel-wise binary classification tasks. We archieved state-of-the-art performace in edge detection and road extraction. We think Ada-LSN is also suitable for other tasks, for example, in subsequent experiments, we found Ada-LSN also works well on building extraction. You can use our method to simply modify the data path and run:
```
python train_AdaLSN.py
```
Please refer to ODN[<sup>6</sup>](#odn) for saliency detection and earthquake detection 
## Reference
<div id="hed"></div>
- [1] S. Xie and Z. Tu, “Holistically-nested edge detection,” in IEEE ICCV, 2015
<div id="srn"></div>
- [2] W. Ke, J. Chen, J. Jiao, G. Zhao, and Q. Ye, “SRN: side-output residual network for object symmetry detection in the wild,” in IEEE CVPR,2017.
- [3] K. Zhao, W. Shen, S. Gao, D. Li, and M. Cheng, “Hi-fi: Hierarchical feature integration for skeleton detection,” in IJCAI, 2018.
<div id="deepflux"></div>
- [4] Y. Wang, Y. Xu, S. Tsogkas, X. Bai, S. J. Dickinson, and K. Siddiqi, “Deepflux for skeletons in the wild,” in IEEE CVPR, 2019.
<div id="adalsn"></div>
- [5] Chang Liu*, Yunjie Tian*, Jianbin Jiao and Qixiang Ye, "Adaptive Linear Span Network for Object Skeleton Detection", https://arxiv.org/abs/2011.03972.
<div id="adalsn"></div>
- [6] Chang Liu, Fang Wan, Wei Ke, Zhuowei Xiao, Yuan Yao, Xiaosong Zhang and Qixiang Ye, "Orthogonal Decomposition Network for Pixel-Wise Binary Classification", in IEEE CVPR, 2019.
- [7] Chang Liu, Wei Ke, Fei Qin and Qixiang Ye, "Linear Span Network for Object Skeleton Detection", in IEEE ECCV, 2018.

5.SELF-SUPERVISED VARIATIONAL AUTO-ENCODERS
# VAE and Super-Resolution VAE in PyTorch
[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB.svg?logo=python)](https://www.python.org/) [![PyTorch 1.3](https://img.shields.io/badge/PyTorch-1.3-EE4C2C.svg?logo=pytorch)](https://pytorch.org/docs/1.3.0/) [![MIT](https://img.shields.io/badge/License-MIT-3DA639.svg?logo=open-source-initiative)](LICENSE)
Code release for [Super-Resolution Variational Auto-Encoders](https://arxiv.org/abs/2006.05218)
<p align="center">
  <img src="readme_imgs/model_graphs.png" width="500"/>
</p>
### Abstract
<em>The framework of Variational Auto-Encoders (VAEs) provides a principled manner of reasoning in latent-variable models using variational inference.
However, the main drawback of this approach is blurriness of generated images.
Some studies link this effect to the objective function, namely, the (negative) log-likelihood function.
Here, we propose to enhance VAEs by adding a random variable that is a downscaled version of the original image and still use the log-likelihood function as the learning objective.
Further, we provide the downscaled image as an input to the decoder and use it in a manner similar to the super-resolution.
We present empirically that the proposed approach performs comparably to VAEs in terms of the negative log-likelihood function, but it obtains a better FID score.</em>
## Features
- __Models__
    * VAE
    * Super-resolution VAE (srVAE)
- __Priors__
    * Standard (unimodal) Gaussian
    * Mixture of Gaussians
    * RealNVP
- __Reconstruction Loss__
    * Discretized Mixture of Logistics Loss
- __Neural Networks__
    * DenseNet
- __Datasets__
   * CIFAR-10
## Quantitative results
<center>
| **Model**  |   **nll**   |
| :---   |  :---:  |
| VAE    |  3.51   |
| srVAE  |  3.65   |
Results on CIFAR-10. The log-likelihood value *nll* was estimated using 500 weighted samples on the test set (10k images).
</center>
## Qualitative results
### VAE
Results from VAE with RealNVP Prior trained on CIFAR10.
<!-- Interpolations -->
<p align="center">
  <img src="readme_imgs/vae_inter1.jpg" width="500" align="center"/>
  <img src="readme_imgs/vae_inter2.jpg" width="500" align="center"/>
</p>
<p align="center">
    Interpolations
</p>
<!-- Reconstructions -->
<p align="center">
  <img src="readme_imgs/vae_recon.jpg" width="500" />
</p>
<p align="center">
    Reconstructions.
</p>
<!-- Generations -->
<p align="center">
  <img src="readme_imgs/vae_generations.jpg" width="400" />
</p>
<p align="center">
    Unconditional generations.
</p>
### Super-Resolution VAE
Results from Super-Resolution VAE trained on CIFAR10.
<!-- Interpolations -->
<p align="center">
  <img src="readme_imgs/svae_inter1.jpg" width="500" align="center"/>
  <img src="readme_imgs/svae_inter2.jpg" width="500" align="center"/>
</p>
<p align="center">
    Interpolations
</p>
<!-- Super-Resolution -->
<p align="center">
  <img src="readme_imgs/srvae_super_res.jpg" width="500" />
</p>
<p align="center">
    Super-Resolution results of the srVAE on CIFAR-10
</p>
<!-- Generations -->
<p align="center">
  <img src="readme_imgs/srvae_generations_y.jpg" width="400" />
  <img src="readme_imgs/srvae_generations_x.jpg" width="400" />
</p>
<p align="center">
    Unconditional generations.
    <b>Left:</b> The generations of the first step, the compressed representations that capture the _global_ structure.
    <b>Right:</b> The final result after enhasing the images with local content.
</p>
## Requirements
The code is compatible with:
  * `python 3.6`
  * `pytorch 1.3`
## Usage
 - To run VAE with RealNVP prior on CIFAR-10, please execude:
```
python main.py --model VAE --network densenet32 --prior RealNVP
```
 - Otherwise, to run srVAE:
```
python main.py --model srVAE --network densenet16x32 --prior RealNVP
```
## Cite
Please cite our paper if you use this code in your own work:
```
@misc{gatopoulos2020superresolution,
    title={Super-resolution Variational Auto-Encoders},
    author={Ioannis Gatopoulos and Maarten Stol and Jakub M. Tomczak},
    year={2020},
    eprint={2006.05218},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
#### *Acknowledgements*
<em>This work was supported and funded from the [University of Amsterdam](https://www.uva.nl/), and [BrainCreators B.V.](https://braincreators.com/)</em>.
#### *Repo Author*
Ioannis Gatopoulos, 2020

6.“SMED.m” is the edge detector based on scale multiplication.Refer to P. Bao, L. Zhang, and X-L Wu, “Canny edge detection enhancement by scalemultiplication,” IEEE Trans. Pattern Anal. Mach. Intell., 27(9): 1485-1490, 2005.

7.“anisotropic_Directional_derivative_filter.m” is used to a set of anisotropic directional derivative filters. The spatial support of the filter is [-20,20]×[-20,20] and the orientation angles are uniformly distributed on the interval [0,π].

8.“non_maxima_suppression.m” is used to extract the maxima of the gradient magnitude of an image by using the two partial derivatives of the image. Refer toM. S. Nixon and A. S. Aguado, “Feature Extraction and Image Processing,” Chapter 3, Newnes, 2002.

@ARTICLE{9706179,  
author={Yu, Xiaohang and Wang, Xinyu and Liu, Jie and Xie, Rongrong and Li, Yunhong},  
journal={IEEE Access},   
title={Multiscale Anisotropic Morphological Directional Derivatives for Noise-Robust Image Edge Detection},   
year={2022},  
volume={10},  
number={}, 
pages={19162-19173},  
doi={10.1109/ACCESS.2022.3149520}}

