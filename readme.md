# Deepfake Detection

A project by William Convertino, Yunrong Cai, and Jake Wolfram

### Dataset Installation

To install the dataset,

## Usage

### 1. Train a Model

An example for resnet18:

```bash
python main.py --train resnet18
```

### 2. Evaluate a Model

To evaluate the model at a specific checkpoint (ex. epoch 5):

```bash
python main.py --eval resnet18 --epoch 5_epoch
```

### 3. Visualize Model Training

An example for resnet18:

```bash
python main.py --visualize resnet18
```

### Usage Notes

To start training from a specific epoch, use:
