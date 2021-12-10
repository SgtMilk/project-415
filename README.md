# project-415
The github repo of the ECSE-415 project

![Alt text](CPU_train.png?raw=true "CPU slavery")


The project is divided into 3 parts, one Jupyter notebook per question. 

We have provided a virtual environment for this project. To activate it, you need the `virtualenv` package, and will need to type `source project/bin/activate`. Inside of the virtual environment, type `pip install -r requirements.txt` to install all the required packages.

# RUNNING Q2.ipynb
The training pictures have to be in the `./data/TRANCOS` forder and the .csv files in the `./data` folder. We recommend running Q2 on a CUDA-enabled GPU, as it would take a long time on CPU.

# RUNNING Q3_b.ipynb
Place the training data in a subfolder named /data/original. This folder should directly contain the images given. The folder data should contain the csv for the test, train and valid set.
Run all the Python code boxes for training the regressor and creating an output csv file with the testset results.
