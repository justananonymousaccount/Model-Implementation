# LIMD

[Project](https://justananonymousaccount.github.io/Do-More-with-Less/) | [Paper](https://drive.google.com/file/d/1NVQ1FTtc44RwXbXcjjsHOYC3LG2LvtlF/view) | [Supplementary materials](https://matankleiner.github.io/sinddm/resources/sinddm_supp.pdf)
### Official pytorch implementation of the paper: "Do More with Less: Pushing Boundaries in Zero-Shot Super-Resolution with Diffusion Models"

## Table of Contents

* [Abstract](#Abstract)
* [Requirements](#Requirements)
* [Training](#Training)
* [Inference and Super Resolution](#Inference-and-Super-Resolution)
* [Data and Pretrained Models](#Data-and-Pretrained-Models)
* [Sources](#Sources)

## Abstract
---
>Super-resolution (SR) techniques have made significant strides with the advent of diffusion models, yet these methods are often data-hungry, computationally intensive, and struggle with real-world degradations. In this work, we propose a novel zero-shot super-resolution framework that leverages diffusion models without the need for extensive datasets. Our approach addresses the limitations of current diffusion-based models, particularly their reliance on heavy architectures like UNet, which require significant computational resources and long training times. By adopting a fully convolutional ConvNeXt architecture, we eliminate the need for attention layers, thus reducing memory overhead and speeding up training, while still achieving results with high fidelity and perceptual quality. We introduce a multi-scale training strategy using a single image, where pseudo high-resolution (HR) and low-resolution (LR) pairs are generated through a combination of unsharp masking and controlled degradation. The proposed training pipeline involves step-wise tuning across scales, enabling efficient convergence. Unlike traditional approaches that are limited to fixed upscaling factors, our method supports high-quality arbitrary scale super-resolution by gradually increasing resolution during inference. Our model achieves competitive results in real-world scenarios with complex degradations, advancing zero-shot super-resolution with minimal data and computation.
---

## Requirements
A suitable [conda](https://conda.io/) environment named `limd` can be created and activated with:
```
conda create -n limd python=3.10
conda activate limd
pip install -r requirements.txt
```

## Training
To train a LIMD model on your own image e.g. `<training_image.png>`, put the desired training image under `./datasets/`, and run

```
python main.py --scope <training_image> --mode train --dataset_folder ./datasets/ --image_name <training_image.png> --results_folder ./results/ 
```

## Inference and Super Resolution 
To generate super resolution result on scale factor `r`, please first train a LIMD model on the desired image (as described above) or use a provided pretrained model, then run

```
python main.py --scope <training_image> --mode SR --dataset_folder ./datasets/ --image_name <training_image.png> --results_folder ./results/ --sr_factor r --load_milestone 4
```

## Data and Pretrained Models
We provide several pre-trained models for you to use under `./results/` directory. More models will be available soon.
 
We provide all the training images we used in our paper under the `./datasets/` directory. All the images we provide are in the dimensions we used for training and are in .png format. 

## Sources 
The diffusion process code was adapted from the [ResShift](https://github.com/zsyOAOA/ResShift/tree/journal) and the degradation pipeline is inspired from [BSRGAN](https://github.com/cszn/BSRGAN)





