﻿# INM706 Image Captioning
 
 This is a sequence analysis project for INM706 model in MSc Artificial Intelligence course at [City University of London](https://www.city.ac.uk/).
 
 This project is based on **Show and Tell: A Neural Image Caption Generator.** *Oriol Vinyals, et al.* CVPR 2015. you can find the paper [here](https://arxiv.org/pdf/1411.4555.pdf).

&nbsp;

## Requirements
First, make sure your environment is installed with Python >= 3.5
Then install requirements:

```bash
pip install -r requirements.txt
```

&nbsp;

## Dataset
You used MS COCO dataset 2017 version. you can dowload it fom their [website](https://cocodataset.org/#download).
Or you can use the commad line:
```bash
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
```
and for annotations:
```bash
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```
