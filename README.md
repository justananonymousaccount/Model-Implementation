# Escaping the Mirage

[Project](https://justananonymousaccount.github.io/Authentic-Super-Resolution/) | [Paper](https://justananonymousaccount.github.io/Authentic-Super-Resolution/) | [Supplementary Material](https://justananonymousaccount.github.io/Authentic-Super-Resolution/) | [Colab](https://colab.research.google.com/drive/1y3RYi8tD_95z0jD-d2E0hHb9y5WXAMvk?usp=sharing)
### Official pytorch implementation of the paper: "Escaping the Mirage: Realizing Authentic Super Resolution with Scale-Aware Zero-Shot Diffusion"

## Table of Contents

* [Abstract](#Abstract)
* [Requirements](#Requirements)
* [Training](#Training)
* [Inference and Super Resolution](#Inference-and-Super-Resolution)
* [Data and Pretrained Models](#Data-and-Pretrained-Models)
* [Sources](#Sources)

## Abstract
---
>Regardless of advancements in supervised super-resolution techniques, the pursuit of generalization often comes at the cost of hallucinations. Moreover, even after days of training on expensive computational resources, these models struggle with images containing degradations outside their training distribution. The rise of self-supervised super-resolution techniques addresses these challenges through internal learning, where the model is trained directly on a single test image. Our approach further overcomes the limitations of existing self-supervised methods by leveraging the diffusion process to achieve high-fidelity reconstructions with enhanced perceptual quality. We first utilize a patch-based training framework to satisfy the data hunger of diffusion models and prevent overfitting. Then, we propose a lightweight ConvNeXt-V2 based UNet for denoising architecture. Additionally, incorporating Discrete Wavelet Transform (DWT) and Fast Fourier Transform (FFT) based combined loss enhances our self-supervised model's emphasis on high-frequency details.
---

## Requirements
A suitable [conda](https://conda.io/) environment named `limd` can be created and activated with:
```
conda create -n limd python=3.10
conda activate limd
pip install -r requirements.txt
```

## Training
To train a model on your own image e.g. `<training_image.png>`, put the desired training image under `./datasets/`, and run

```
python main.py --scope <scope_name> --mode train --dataset_folder ./datasets/ --image_name <training_image.png> --results_folder ./results/
```

## Inference and Super Resolution 
To generate super resolution result for a given resolution, please first train a model on the desired image (as described above) or use a provided pretrained model, then run

```
python main.py --scope <scope_name> --mode SR --dataset_folder ./datasets/ --image_name <training_image.png> --results_folder ./results/ --widthl 2 4 --heightl 2 4 --load_milestone 2
```

## Data and Pretrained Models
We will provide some pre-trained models for you to use under `./results/` directory.
 
We will provide test images we used in our paper under the `./datasets/` directory. All the images we provide are in the dimensions we used for training and are in .png format. 

## Sources 
The diffusion process code is adapted from the [ResShift](https://github.com/zsyOAOA/ResShift/tree/journal) and the degradation pipeline is inspired from [BSRGAN](https://github.com/cszn/BSRGAN).





