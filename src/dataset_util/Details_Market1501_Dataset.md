## Details about Market-1501 dataset

## Dataset Referencing:

@inproceedings{zheng2015scalable,
title={Scalable Person Re-identification: A Benchmark},
author={Zheng, Liang and Shen, Liyue and Tian, Luand Wang, Shengjin and Wang, Jingdong and Tian,
Qi},
booktitle={Computer Vision, IEEE International Conferenceon},
year={2015}
}

## PDF:

https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identifiation_ICCV_2015_paper.pdf

## ** Note Table 1 compares Market-1501 dataset to other existing dataset

## The Market-1501 dataset three featured properties(3.1. Description Paper):

1. Dataset uses the Deformable Part Model (DPM) as a pedestrian detector.
2. In addition to the true positive bounding boxes, we also provide false alarm detection results.
3. Each identity may have multiple images under each camera. During cross-camera search, there are multiple queries and multiple ground truths for each identity.

## Test and training sets (3.3. EvaluationProtocol Paper):

The dataset is randomly divided into training and testing sets, containing 750 and 751 identities,
respectively. During testing, for each identity, selected one query image in each camera. Note that,
● The selected queries are hand-drawn, instead of DPM-detected as in the gallery.
● The reason is that in reality, it is very convenient to interactively draw a box, which can yield higher
recognition accuracy [20].
● [20] : W. Li, R. Zhao, T. Xiao, and X. Wang. Deep reid: Deep filter pairing neural network for person
re-identification. In CVPR, pages 152– 159, 2014.

## Feature Extraction (4.1. The Bag-of-Words Model Paper)

A classic BoW model is constructed. Used dense sampling and extract a 11-dim Color Names [8] vector for
each patch.
The descriptor is l1-normalized followed by square root operator [10]. A codebook is trained on the
irrelevant TUD-Brussels dataset [9].
Then, a given feature vector is quantized to its nearest neighbor under Euclidean distance. Employed
Multiple Assignment (MA) [11] and set MA = 10.
Moreover, also integrate Burstiness weighting [12]and Negative Evidence [13] into the BoW model.

## Performance Evaluation (http://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)

In this implementation, using the following components:
A. baseline: bow descriptor + linear scan. You will obtainmAP = 14.75%, r1 accuracy = 35.84%;
B. baseline + multiple query by max/avg pooling;
C. baseline + multiple query by max-pooling + re-ranking;
D. baseline + pairwise evaluation: re-id between camera pairs is evaluated, and a confusion matrix is
drawn.
E. metric learning: Using the code provided by [14], we provide the training and testing protocol on our
dataset.

## The Market-1501 dataset annotated using the following rules: (http://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)

● For each detected bounding box to be annotated, wemanually draw a ground truth bounding box
that contains the pedestrian.
● Then, for the detected and hand-drawn bounding boxes,we calculate the ratio of the overlapping
area to the union area.
● If the ratio is larger than 50%, the DPM bounding box is marked as "good"; if the ratio is smaller than
20%, the bounding boxe is marked as "distractor";otherwise, it is marked as "junk", meaning that
this image is of zero influence to the re-identificationaccuracy.
