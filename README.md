# S2MIS-SMFLP-MVC

## Overview

S2MIS-SMFLP-MVC is a deep learning project designed for **semi-supervised medical image segmentation**, as discussed in the paper *"Enhancing Consistency Regularization in Semi-Supervised Medical Image Segmentation: A Posterior Perturbation Perspective"*. The project adopts a semi-supervised learning framework with enhanced consistency regularization to improve segmentation performance. Currently, the project is in a demo phase, and the final version will be released upon paper acceptance. Stay tuned! 😊

## Supported Datasets

**(i) Left Atrium (LA) Dataset:**  
A benchmark dataset for the *2018 Atrial Segmentation Challenge* \url{http://atriaseg2018.cardiacatlas.org}, this dataset includes 100 gadolinium-enhanced **T1-weighted MRI** scans, each with an isotropic resolution of $0.625 \times 0.625 \times 0.625$ mm, providing detailed anatomical visualization of the left atrium. Following the data division and preprocessing protocol outlined in BCP \citep{bai2023bidirectional}, 80 samples are used for training, and the remaining 20 are used for testing.

**(ii) Pancreas-CT Dataset:**  
A widely used dataset for abdominal imaging url{https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT}, this dataset includes 82 contrast-enhanced **3D-CT** scans with a resolution of $512 \times 512$ pixels and slice thicknesses ranging from $1.5$ to $2.5$ mm. Following the data split protocol in MC-Net+ \citep{wu2022mutual}, 62 samples are used for training, and 20 for testing. The images are clipped to a voxel value range of $[-125, 275]$ HU, resampled to an isotropic resolution of $1.0 \times 1.0 \times 1.0$ mm, and normalized to zero mean and unit variance.

**(iii) BraTS-2019 Dataset:**  
The BraTS dataset \url{https://www.med.upenn.edu/cbica/brats2019} contains preoperative MRI scans (modalities: T1, T1Gd, T2, and T2-FLAIR) from 335 glioma patients. In this study, we use **T2-FLAIR** images resampled to an isotropic resolution of $1.0 \times 1.0 \times 1.0$ mm. Following the data split and preprocessing protocol from MRP \citep{su2024mutual}, 250 samples are used for training, 25 for validation, and 60 for testing.

## Important Notice

**Note**: This is a demo version of the code. The final version will be released once the paper is accepted.

## Environment Setup

Ensure the following environment for smooth execution:

- **Operating System**: Linux / Windows / macOS
- **Python Version**: 3.8+
- **CUDA Version**: 10.2+ (if using GPU)

## Required Dependencies

The following Python packages are required for the project:

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
# Additional dependencies may be required depending on the specific dataset and model configurations

## Contact Information
If you have any questions or suggestions about this project, please contact me through email: `zhanghongyu22@mails.jlu.edu.cn` or QQ Group (Chinese): `906808850`.

