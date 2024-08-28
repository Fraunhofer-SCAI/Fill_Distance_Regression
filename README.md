# On minimizing the training set fill distance in machine learning regression
This repository contains the code and experiments to replicate the results presented in the paper titled "[On minimizing the training set fill distance in machine learning regression](https://arxiv.org/pdf/2307.10988.pdf)"

**Paolo Climaco and Jochen Garcke**\
Institut für Numerische Simulation, Universität Bonn, Germany\
Fraunhofer SCAI, Sankt Augustin, Germany

Contact climaco@ins.uni-bonn for questions about code and data.

## Python Packages
-Python (>= 3.7)\
-Pytorch 1.11.0\
-Install packages in requirements.txt


## Repository Structure

```plaintext
.
├── datasets/                   # Folder containing code to access data. Used also for data storage. 
│   ├── Datasets_Class.py           # Code for dowloading and reading datasets.
│    
├── notebooks/                  # Folder containing jupiter notebooks.
│   ├── experiments_QM7,ipynb       # Jupyter Notebook replicating experiments on QM7.
│   ├── access_data.ipynb           # Jupyter Notebook explaining how to access preprocessed datasets.
│
├── Passive_Sampling/           # Folder containing code to select datapoint with FPS.
│   ├──farthest_point.py            # Code for implementing the  Fartehst Point Sampling (FPS).
│
├── utils/                      # Folder containing basic code to run and plot experiments.
│   ├──FNN.py                       # Code containg the FNN architecture, training and testing procedures.
│   ├──plots.py                     # Code plotting the result of the experiments.

└── README.md                   # Project README file.
└── requirements.txt            # python packages required to run code.
```

