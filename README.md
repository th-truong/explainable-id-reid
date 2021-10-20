# Authors
* Divyansh Goyal
* Netra Patel
* Thomas Truong
* Svetlana Yanushkevich

# Introduction
This repository contains a practical and useable model and its associated files to perform person re-identification through attribute recognition.
The model is featured in the scientific paper, _Towards Explainable Person Re-Identification_, which has been accepted for publication and presentation at the __IEEE Symposium Series on Computational Intelligence (SSCI) 2021__.
<br><br>The paper's abstract is:
> Visually recognizing an individual in a crowded area using a distributed camera network is essential for a range of biometric and security applications. We propose a shift in perspective of the ongoing re-identification studies, towards creating more explainable and coherent models that are applicable in real-world engineering problems, even if this comes with a slight decrease in performance. The proposed explainable model uses attribute classification to perform the task of re-identification. This method steps away from  intrusive and controversial techniques such as facial recognition to improve public acceptance of re-identification models. Current methods of person re-identification do not explain the importance of each attribute in determining the results, and often use complicated and esoteric algorithms to improve the performance on closed-world datasets which may not represent more realistic open-world scenarios. We applied our approach to the Market-1501 dataset and examined the impact of careful selection of backbone outputs for each individual attribute in our experiment. Our simple model is capable of performing attribute classification for 0-shot re-identification that is explainable and less intrusive when
compared to state-of-the-art models focused on re-identification.

# Network Architecture
We have applied a ResNet-50/FPN backbone which provides five different output heads, "0", "1", "2", "3", and "pool". Then, we feed in these outputs to a linear classification layer system, which provide attribute predictions for the probe image.
![Network Architecture](https://user-images.githubusercontent.com/58268240/138032890-6f8d315d-1d4a-46de-96f5-2fa7bca0c9b7.PNG)

# Libraries Used
* argparse
* torch
* confuse
* pathlib
* torchvision
* slearn.metrics
* numpy
* tqdm
* pandas
* os
* sys
* matplotlib
* albumentations
* scipy.io
* PIL

# How to Run
To run, install all required libraries and download dataset and update path files in dataset_utils/market1501.yml and model_utils/model_paths.yml, as well as configure the parameters of training and testing in model_util/classifier_architecture.yml. Then, run 'python main.py' when in the src directory and follow the instructions on the terminal.
