## Details about Market-1501 dataset ##
The original dataset contains 751 identities for training and 750 identities for testing. The attributes are annotated in the identity level, thus the file contains 28 x 751 attributes for training and 28 x 750 attributesfor test, where the label "image_index" denotes the identity. 

The dataset package contains four folders.
1. "bounding_box_test". There are 19,732 images in this folder used for testing.
2. "bounding_box_train". There are 12,936 images in this folder used for training.
3. "query". There are 750 identities. We randomly select one query image for each camera. So the maximum number of query images is 6 for an identity. In total, there are 3,368 query images in this folder.
4. "gt_query". This folder contains the ground truth annotations. For each query, the relevant images are marked as "good" or "junk". "junk" has zero impact on search accuracy. "junk" images also include those in the same camera with the query.
5. "gt_bbox". We also provide the hand-drawn bounding boxes. They are used to judge whether a DPM bounding box is good.


## Dataset Referencing:

@inproceedings{zheng2015scalable,
title={Scalable Person Re-identification: A Benchmark},
author={Zheng, Liang and Shen, Liyue and Tian, Luand Wang, Shengjin and Wang, Jingdong and Tian,
Qi},
booktitle={Computer Vision, IEEE International Conferenceon},
year={2015}
}


## The Market-1501 dataset three featured properties(3.1. Description Paper):

Overall, the dataset contains 32,668 annotated bounding boxes of 1,501 identities. In this open system, images of each identity are captured by at most six cameras. Made sure that each annotated identity is present in at least two cameras, so that cross-camera search can be performed. The Market-1501 dataset has three featured properties:
1. Dataset uses the Deformable Part Model (DPM) as a pedestrian detector.
2. In addition to the true positive bounding boxes, we also provide false alarm detection results.
3. Each identity may have multiple images under each camera. During cross-camera search, there are multiple queries and multiple ground truths for each identity.

## Test and training sets (3.3. EvaluationProtocol Paper):

The dataset is randomly divided into training and testing sets, containing 750 and 751 identities,
respectively. During testing, for each identity, selected one query image in each camera. Note that,
* The selected queries are hand-drawn, instead of DPM-detected as in the gallery.
* The reason is that in reality, it is very convenient to interactively draw a box, which can yield higher
recognition accuracy [20].
* [20] : W. Li, R. Zhao, T. Xiao, and X. Wang. Deep reid: Deep filter pairing neural network for person
re-identification. In CVPR, pages 152– 159, 2014.


## The Market-1501 dataset annotated using the following rules: (http://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)

* For each detected bounding box to be annotated, manually draw a ground truth bounding box
that contains the pedestrian.
* Then, for the detected and hand-drawn bounding boxes, calculate the ratio of the overlapping
area to the union area.
* If the ratio is larger than 50%, the DPM bounding box is marked as "good"; if the ratio is smaller than
20%, the bounding boxe is marked as "distractor";otherwise, it is marked as "junk", meaning that
this image is of zero influence to the re-identification accuracy.

## Other results from papers
### Attribute Classification:
* ![image](https://user-images.githubusercontent.com/58268240/122836687-efd8ba80-d2af-11eb-8b6a-e6c0bf00daed.png)
@unknown{unknown,
author = {Liu, Hao and Wu, Jingjing and Jiang, Jianguo and Qi, Meibin and Bo, Ren},
year = {2018},
month = {11},
pages = {},
title = {Sequence-based Person Attribute Recognition with Joint CTC-Attention Model}
}

