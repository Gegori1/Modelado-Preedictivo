## HOMEWORK_NN

![Neural Networks is more art than science](nn_sketch.png)

### Description

This is a repository containing 1 Predictive Modeling homework. The problem at hand is to implement a neural network with the keras library, to solve a regression problem [Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction).

The repository contains the following files:

- `Readme.md`: this very file, describing the repository content and the homework.

- `makefile`: a makefile containing the commands to run the code.

- `data`: a folder containing the dataset used for the homework.

- `Modeling`: a folder containing the jupyter notebook with the code for the homework.

- `Report`: a folder containing the report of the homework.

- `Grid_Search`: a folder containing the jupyter notebook with the code for the grid search.

- `P9_MLP_.py`: a python script containing an implementation of a neural network with the keras library for a different problem.

- `Data_Analysis.ipynb`: a jupyter notebook containing the data analysis for the homework.

- `sdk`: a folder containing the sdk for the homework.

### The repository is organized as follows

```
├── README.md
├── makefile
├── data
│   ├── energydata_complete.csv
│   └── 
├── Modeling
│   └── linear_model.ipynb
│   └── NN_model.ipynb
├── Grid_Search
│   └── grid_search_test.ipynb
├── P9_MLP_.py
├── Data_Analysis.ipynb
├── Report
│   └── Report__NN_Regression.md
│   └── Report__NN_Regression.pdf
└── sdk
    ├── neural_networks
    │   ├── __init__.py
    │   ├── Model.py
    │   ├── nn_utils.py
    ├── setup.py
    ├── setup.cfg
    └── pyproject.toml
    └── requirements.txt
```

### Instructions to run the code

To run the code, a python version >= 3.7 is required. It is recommended to create a virtual environment and install the required packages.
To use the make commands it is necessary to have make installed, which can be install on Linux with the command:

```bash
sudo apt install make
```

or in windows downloading the binaries through this [link](http://gnuwin32.sourceforge.net/packages/make.htm) or using the [chocolatey](https://chocolatey.org/) package manager with the command:

```bash
choco install make
```

1. Clone the repository with ssh

```bash
#! using ssh
git clone git@github.com:Gegori1/Modelado-Preedictivo.git
#! using https
git clone https://github.com/Gegori1/Modelado-Preedictivo.git
```

2. Move to the repository folder

```bash
cd Modelado-Preedictivo
```

4. Move to the HOMEWORK_NN folder

```bash
cd HOMEWORK_NN
```

5. Create a virtual environment

```bash
# using makefile
make create_env
# using python
python -m venv env
```

6. Activate the virtual environment

```
# windows cmd
env\Scripts\activate.bat
# linux
source env/bin/activate
```

7. Install the required packages

```bash
# using makefile
make install_requirements
# using pip
pip install -r ./sdk/requirements.txt
```

8. Install local sdk

```bash
# using makefile
make install_local
# using pip
pip install -e ./sdk
```

9. Run the code

Pending...