## Introduction

This is the code repo for paper "Modular Deep-Learning-Based Early Warning System for Deadly Heatwave Prediction". The repo contains: 1. Codes for the model 2. Sample data used to verify the model 

## Execution

To run the model, use the following command:

```
python3 main.py <Year> <City Code> <Data Subgroup> <Local Device>
```

Here, `Year` represents the year model trying to predict; `City Code` and `Data Subgroup` are identifiers inside the preprocessed data; `Local Device` is the identifier of GPU device where the script will be executed. A example command is:

```
python3 main.py 2000 AAA ine cuda:0
```

## Data

An anonymized sample data is provided in `./data`. 
