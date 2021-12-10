# Project-415 Vehicle Detection and Counting
The github repo of the ECSE-415 project

![Alt text](CPU_train.png?raw=true "CPU slavery")


The project is divided into 3 parts, one Jupyter notebook per question. 

# Q1.ipynb
Q1 has been developped on Google Collab.
## Run Instructions
### Path Variables

The path variables are defined as such:


```
csv_path = '/content/drive/My Drive/University/ECSE-415/ECSE_415_F_2021_Project/vehiclecounting/valid_count.csv'
path = '/content/drive/My Drive/University/ECSE-415/ECSE_415_F_2021_Project/vehiclecounting/TRANCOS/TRANCOS/'
```

To add your own path, simply replace "path" to the desired directory you have.

### Libraries Used
```
import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import torch
!pip install pyyaml==5.1
from google.colab import drive
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import glob
drive.mount('/content/drive')
import pandas as pd
from scipy import stats'
```

## References
I used the internet to help me debug some issues I encountered. Here are all the links:
* https://github.com/facebookresearch/detectron2/blob/master/.github/ISSUE_TEMPLATE/unexpected-problems-bugs.md
* https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=HUjkwRsOn1O0
* https://detectron2.readthedocs.io/en/latest/modules/model_zoo.html
* https://github.com/facebookresearch/detectron2/issues/1277
* https://github.com/facebookresearch/detectron2/issues/1962
* https://towardsdatascience.com/non-maxima-suppression-139f7e00f0b5
* https://www.dataquest.io/blog/understanding-regression-error-metrics/

# Run Instruction Q3_b.ipynb
Place the training data in a subfolder named /data/original. This folder should directly contain the images given. The folder data should contain the csv for the test, train and valid set.
Run all the Python code boxes for training the regressor and creating an output csv file with the testset results.
