# Computer Vision Vehicle Detection and Counting

![Alt text](CPU_train.png?raw=true "CPU slavery")


The project is divided into 3 parts, one Jupyter notebook per question. 

We have provided a virtual environment for this project. To activate it, you need the `virtualenv` package, and will need to type `source project/bin/activate`. Inside of the virtual environment, type `pip install -r requirements.txt` to install all the required packages. Note that for Q1 google colab was used, so it does not use that virtual environment. We have specified the needed packages below.

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
* https://towardsdatascience.com/non-maxima-suppression-139f7Place the training data in a subfolder named /data/original. This folder should directly contain the images given. The folder data should contain the csv for the test, train and valid set.
Run all the Python code boxes for training the regressor and creating an output csv file with the testset results.
The training pictures have to be in the `./data/TRANCOS` forder and the .csv files in the `./data` folder. We recommend running Q2 on a CUDA-enabled GPU, as it would take a long time on CPU.
Running the code is simply done by pressing the `Run all` button while in the virtual environment and takes around an hour on a RTX 2060 super.

# Run Instruction Q3_b.ipynb
The question uses the same data as Q2. Make sure the csv files are in the data folder. Run all the Python cells to train the regressor on the training set, validate it against the validation set and finally output a csv file for a kaggle submission. 

## References
I used the following references to develop this part of the question:
* https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
* https://www.thepythoncode.com/article/hog-feature-extraction-in-python
* https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
* https://scikit-learn.org/stable/modules/svm.html#svm-regression
