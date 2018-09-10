# Implementation of a Neural Network with Backpropagation Algorithm

University Project for Machine Learning course. 

This is an implementation of a Neural Network with the Backpropagation Algorithm, using Momentum and L2 Regularization.

## Getting Started

To download my repo:

```
git clone https://github.com/riki95/Neural-Network-Backpropagation-ML-Project
```

The program was written and executed with [Matlab](https://it.mathworks.com/downloads/).

The dataset used are Monks for Classification and Wines Quality for Regression, but you can modify the launch files to use different datasets.

### Monks - Classification

[Monks Dataset](https://archive.ics.uci.edu/ml/datasets/MONK%27s+Problems) was downloaded and modified in order to get only a file instead of 2 different files for train and validation.

You can find the Monks Dataset inside the Data folder, starting from the first file which is the easier to the last one which is the harder to train.

### Wine Quality - Regression

[Wine Quality Dataset](http://archive.ics.uci.edu/ml/datasets/Wine+Quality
) was downloaded and it's ready to be used, you can find it on the Data folder.

The files are 2, one for Red Wines and one for White Wines.


## Running the tests

Just open the project with Matlab and run the "LaunchMonks" or "LaunchWines" files. There are 2 files from both, depending on the type of algorythm to use for Validation: Holdout or K-Fold.

Inside the launch files you can also set some parameters, for example "Validation" if you want to use it or not.

## Authors

* **Riccardo Basso** - *S4071408* - Università degli studi di Genova