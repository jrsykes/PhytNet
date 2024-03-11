# Import necessary libraries and modules
import torch
from PhytNet import PhytNetV0  # Assuming a custom model defined elsewhere
import wandb  # For logging and hyperparameter sweeps with Weights & Biases
import yaml  # To handle YAML files, e.g., for configuration
import pprint  # Pretty print for more readable output of data structures
import os  # For operating system dependent functionality, e.g., file paths
from torchvision import datasets, transforms  # For data loading and transformation

# Set the device to GPU if available, otherwise fallback to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the directory containing your data
data_dir = "/your_data"
# Initialize sweep_config and sweep_id variables
sweep_config = None  # or "./PhytNet-Sweep-config.yml" to run the wandb sweep
sweep_id = None
project_name = "PhytNet-Sweep"  # Name of the project for Weights & Biases
sweep_count = 500  # Number of runs in the sweep



# Example of a config dictionary for a Wandb sweep, values will be provided by Wandb
config = {
    'input_size': wandb.config.input_size,
    'dim_1': wandb.config.dim_1, 
    'dim_2': wandb.config.dim_2, 
    'dim_3': wandb.config.dim_3,
    'kernel_1': wandb.config.kernel_1, 
    'kernel_2': wandb.config.kernel_2,
    'kernel_3': wandb.config.kernel_3,
    'num_blocks_1': wandb.config.num_blocks_1,
    'num_blocks_2': wandb.config.num_blocks_2,
    'out_channels': wandb.config.out_channels,    
    'batch_size': 42,
    'beta1': wandb.config.beta1,
    'beta2': wandb.config.beta2,  
    'learning_rate': wandb.config.learning_rate,
}

# Example of an optimized config dictionary after hyperparameter tuning. Provide such values following your hyperparameter optimization for you dataset
# config = {
#     'beta1': 0.9657828624377116,
#     'beta2': 0.9908102731106424,
#     'dim_1': 104,
#     'dim_2': 109,
#     'dim_3': 110,
#     'input_size': 350,
#     'kernel_1': 5,
#     'kernel_2': 7,
#     'kernel_3': 13,
#     'learning_rate': 0.00013365304940966892,
#     'num_blocks_1': 1,
#     'num_blocks_2': 2,
#     'out_channels': 9,
#     'batch_size': 42,
# }

# Initialize the model with the provided configuration and send it to the specified device
model = PhytNetV0(config=config).to(device)

# Define data transformations for training and validation datasets
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((config['input_size'],config['input_size'])), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(1,3)),
                transforms.RandomRotation(degrees=5)
            ], p=0.4), 
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((config['input_size'],config['input_size'])),
        transforms.ToTensor(),
    ])
}   

# Load datasets with the defined transformations and create dataloaders
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=config['batch_size'], shuffle=True, num_workers=6, drop_last=False) for x in ['train', 'val']}

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], betas=(config['beta1'],config['beta2']))



def train():
    # Placeholder for the training loop
    pass



# Check if a sweep configuration is specified, load it, and run the sweep
if sweep_config != None:
    with open(sweep_config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        sweep_config = config['sweep_config']
        sweep_config['metric'] = config['metric']
        sweep_config['parameters'] = config['parameters']

        # Print the loaded sweep configuration
        print('Sweep config:')
        pprint.pprint(sweep_config)

        # Initialize the sweep or use an existing sweep ID
        if sweep_id is None:
            sweep_id = wandb.sweep(sweep=sweep_config, project=project_name, entity="frankslab")
        else:
            sweep_id = sweep_id
        print("Sweep ID: ", sweep_id)
        print()

    # Start the sweep agent
    wandb.agent(sweep_id, project=project_name, function=train, count=sweep_count)
else:
    # If no sweep configuration is provided, just run the training function
    train()
