# Deep Propensity Model

This repository contains code and scripts for training and evaluating a deep propensity model.

## Setup Instructions

1. **Update Data Path**
    - Navigate to the `dl_data.sh` file.
    - Update the data path to point to your local dataset location.

2. **Update Script Path**
    - Open `run.sh` in the root directory.
    - Edit the script to update any paths so they match your environment.

## Usage

After updating the paths as described above, you can run the data download script:

```bash
sh dl_data.sh
```

Then, run the model training script. It may take a while so you can edit the code to take only a sample:

```bash
sh run.sh
```

## Requirements

- Python 3.x
- There are not a lot of requirements, just install the usual pandas, numpy, and torch according to your machine specs.
