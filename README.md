
# PhytNetV0 Training Framework

This repository contains the implementation for training the PhytNetV0 model, a convolutional neural network designed specifically for classifying plant diseases from images. The training script supports both standard training and hyperparameter optimization through Weights & Biases (WandB) sweeps. The predefined optimisation sweep shoudl be run to this the model archtecture you your dataset before final training. While bolerplate dataloading code etc has been provided, the train() function has been left blank for you to insert your own code.
The original paper for PhytNet can be found here: https://arxiv.org/abs/2311.12088

## Abstract

Automated disease, weed and crop classification with computer vision will be invaluable in the future of agriculture. However, existing model architectures like ResNet, EfficientNet and ConvNeXt often underperform on smaller, specialised datasets typical of such projects. We address this gap with informed data collection and the development of a new CNN architecture, PhytNet. Utilising a novel dataset of infrared cocoa tree images, we demonstrate PhytNet's development and compare its performance with existing architectures. Data collection was informed by analysis of spectroscopy data, which provided useful insights into the spectral characteristics of cocoa trees. Such information could inform future data collection and model development. Cocoa was chosen as a focal species due to the diverse pathology of its diseases, which pose significant challenges for detection. ResNet18 showed some signs of overfitting, while EfficientNet variants showed distinct signs of overfitting. By contrast, PhytNet displayed excellent attention to relevant features, no overfitting, and an exceptionally low computation cost (1.19 GFLOPS). As such PhytNet is a promising candidate for rapid disease or plant classification, or precise localisation of disease symptoms for autonomous systems.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x
- PyTorch
- torchvision
- Weights & Biases (wandb)
- PyYAML

You can install the necessary libraries using pip:

```bash
pip install torch torchvision wandb pyyaml
```

## Configuration

Hyperparameters and training configurations can be adjusted in two ways:

1. Directly in the script by modifying the `config` dictionary.
2. Through a YAML file for WandB sweeps (see `PhytNet-Sweep-config.yml` for an example).

## Dataset

Your dataset should be organized into separate `train` and `val` directories, each containing subdirectories for each class. Place your dataset in a known directory and update the `data_dir` variable in the script accordingly.

## Usage

To run the training script, simply execute:

```bash
python train_phytnet.py
```

To initiate a WandB sweep, first ensure you have a `PhytNet-Sweep-config.yml` file configured with your desired sweep parameters. Then, set `sweep_config` in the script to the path of your YAML file.

## Hyperparameter Tuning

When using WandB for hyperparameter tuning, make sure you're logged into WandB and have configured `PhytNet-Sweep-config.yml` with your desired sweep parameters. The script will automatically handle the sweep initialization and execution.

## Contributing

Contributions to the PhytNet architecture are welcome. Before contributing, please review the contribution guidelines.

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## Acknowledgments

- This project is built using PyTorch and Weights & Biases for model training and hyperparameter optimization.

---
