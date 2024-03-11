
# PhytNetV0 Training Framework

This repository contains the implementation for training the PhytNetV0 model, a deep learning model designed for [insert brief description of the model's purpose, e.g., "classifying plant diseases from images"]. The training script supports both standard training and hyperparameter optimization through Weights & Biases (WandB) sweeps.

## Features

- Training and validation dataset loading and preprocessing
- Model training with customizable hyperparameters
- Optional hyperparameter tuning using WandB sweeps
- GPU/CPU compatibility

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

If you wish to use WandB for hyperparameter tuning, make sure you're logged into WandB and have configured `PhytNet-Sweep-config.yml` with your desired sweep parameters. The script will automatically handle the sweep initialization and execution.

## Contributing

Contributions to the PhytNetV0 training framework are welcome. Before contributing, please review the contribution guidelines.

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## Acknowledgments

- This project is built using PyTorch and Weights & Biases for model training and hyperparameter optimization.

---

Make sure to customize the sections like "[insert brief description of the model's purpose, e.g., 'classifying plant diseases from images']" with information relevant to your project. Adjust any instructions and descriptions according to your project's specific requirements and setup.