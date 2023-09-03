# Weakly-supervised Point Cloud Instance Segmentation from Noisy Bounding Box Annotations via SAM Auto-labeling
- First Author: Qingtao Yu (Research assitant)
- Institutes: Australian National University; University of Queensland

## Abstract
Learning from bounding-boxes annotations has shown great potential in weakly-supervised point cloud instance segmentation. However, we observed that existing methods would suffer severe performance degradation with perturbed bounding box annotations. To tackle this issue, we propose a complementary image prompt-induced weakly-supervised point cloud instance segmentation (CIP-WPIS) method. CIP-WPIS leverages pretrained knowledge embedded in the 2D foundation model [Segment Anything Model](https://github.com/facebookresearch/segment-anything) and 3D geometric prior to achieve accurate point-wise instance labels from the bounding box annotations. Specifically, CP-WPIS first selects projection views in which 3D instance points of interest are fully visible. Then, we generate complementary background and foreground prompts for SAM and employ the SAM heatmap predictions to assign the confidence values for 2D projections in that view. Furthermore, we designed a voting scheme to uniquely assign each point to instances according to the 3D geometric homogeneity of superpoints. In this fashion, we achieved high-quality 3D point-wise instance labels. Extensive experiments on both Scannet-v2 and S3DIS benchmarks demonstrate that our method is robust against noisy 3D bounding-box annotations and achieves state-of-the-art performance.

## Result and Pipline Visualisation

![](./docs/image1.png)
![](./docs/image.png)

*Workflow of our proposed CIP-WPIS. The left part depicts the whole pipeline for obtaining the point-wise instance labels from
noisy bounding boxes. Specifically, we first assign the candidate points for each instance given the noisy bounding boxes. Then we devise
the 3D confidence ensemble module to correct the mislabeled point of each instance. The middle part plots the concrete procedure of the
3D confidence ensemble module. To better exploit the foundation knowledge ensembled in the large 2D foundation model, we first design
a greedy selection algorithm to select multiple 2D views in which an instance is fully visible. Based on projected object points in each 2D
view, we introduce a complementary prompt generation module to obtain the SAM predictions from various views. After that, we integrate
these predictions to indicate whether the point belongs to the instance. The right part details the complementary prompt generation module.
In this module, we introduce the complementary background and foreground prompts to obtain the object mask for each instance.* 


![](./docs/vis2.png)
*Visualisation of results of our CIP-WIPS*

## Installation
```
conda create -n cipwpis
conda activate cipwpis
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install git+https://github.com/facebookresearch/segment-anything.git 
pip install -r requirements.txt
```

## Getting Started

First you need to download Scannetv2 dataset and link it to the main folder.
And see the example commands in run.sh


## Citation
Our arxiv and bibtex will be released soon.

