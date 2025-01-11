# S2MIS-SMFLP-MVC

## Overview

S2MIS-SMFLP-MVC is a deep learning project designed for medical image segmentation, supporting multiple datasets such as Pancreas and LAHeart. This project adopts a semi-supervised learning strategy, combining consistency loss and variational autoencoder (VAE) loss to improve segmentation performance. The project structure is modular, allowing for easy extension and maintenance.

## Important Notice

**Note**: This code version is not the final one, and the final version will be released after acceptance.

## Environment Setup

- **Operating System**: Linux / Windows / macOS
- **Python Version**: 3.8+
- **CUDA Version**: 10.2+ (if using GPU)

## Required Dependencies

Here are the Python packages and their versions required for the project:

```bash
torch==1.10.0
torchvision==0.11.1
tensorboardX==2.4
tqdm==4.62.3
numpy==1.21.2
pandas==1.3.3
opencv-python==4.5.3.56
scikit-learn==0.24.2
matplotlib==3.4.3

Running the Training
To start the training, run the following command in your terminal:

python train.py --dataset_name Pancreas --device cuda --root_path ../ --exp Mine --model mine_v4_test --max_iteration 15000 --max_samples 62 --labeled_bs 2 --batch_size 4 --base_lr 0.01 --deterministic 1 --labelnum 16 --seed 1337 --gpu 0 --consistency 1.25 --consistency_rampup 40.0 --temperature 0.1 --lamda 0.5 --beta 0.5 --N 1


Contact Information
If you have any questions or suggestions about this project, please contact me through email: zhanghongyu22@mails.jlu.edu.cn or QQ Group (Chinese): 906808850.


