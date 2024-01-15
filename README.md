# On minimizing the training set fill distance in machine
This repository contains the code and experiments to replicate the results presented in the paper titled "[On minimizing the training set fill distance in machine](https://arxiv.org/pdf/2307.10988.pdf)"

**Paolo Climaco and Jochen Garcke**\
Institut für Numerische Simulation, Universität Bonn, Germany\
Fraunhofer Center for Machine Learning and SCAI, Sankt Augustin, Germany

Contact climaco@ins.uni-bonn for questions about code and data.

## Python Packages
-Python (>= 3.7)\
-Pytorch 1.11.0\
-Install packeges in requirements.txt


## Repository Structure

```plaintext
.
├── datasets/                   # Folder containg code to access data. Used also for data storage. 
│   ├── Datasets_Class.py           # Code for dowloading and reading datasets.
│    
├── notebooks/                  # Folder containg jupiter notebooks.
│   ├── experiments_QM7,ipynb       # Jupyter Notebook replicating experiments on QM7.
│   ├── access_data.ipynb           # Jupyter Notebook explaining how to access preprocessed datasets.
│
├── Passive_Sampling/           # Folder containg code to select datapoint with FPS.
│   ├──farthest_point.py            # Code for implementing the  Fartehst Point Sampling (FPS).
│
├── utils/                      # Folder containg basic code to run and plot experiments.
│   ├──FNN.py                       # Code containg the FNN architecture, training and testing procedures.
│   ├──plots.py                     # Code plotting the result of the experiments.

└── README.md                   # Project README file.
└── requirements.txt            # python pacakges required to run code.
```
## References
- [1] Jacob Schreiber, Jeffrey Bilmes and William Stafford Noble. apricot: Submodular selection for data summarization in Python. Journal of Machine Learning Research 21(161):1−6, 2020.

