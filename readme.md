# Deepfake Detection

A project by William Convertino, Yunrong Cai, and Jake Wolfram

### Dataset Installation

To install the dataset, run the following commands (linux):

```
chmod +x download.sh
./download.sh
```

### Package Installation

To install relevant packages, run the following command:

```
pip install -r requirements.txt
```

## Usage

### 1. Train a Model

To train a model, use the following command with the name of the config file you wish to train from:

```bash
python main.py --train resnet18
```

To specify a number of epochs (10 by default), use the following command:

```bash
python main.py --train resnet18 --num_epochs 10
```

### 2. Evaluate a Model

To evaluate a model from its best checkpoint:

```bash
python main.py --eval resnet18
```

To evaluate the model at a specific checkpoint (ex. epoch 5):

```bash
python main.py --eval resnet18 --epoch 5_epoch
```

### 3. Visualize Model Training

To produce a training graph from a trained model:

```bash
python main.py --visualize resnet18
```

### Notes

Additional flags (for num_frames, aggregation method, etc.) can be found in main.py
