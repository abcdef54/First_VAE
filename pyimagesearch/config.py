import torch
import os
from torchvision import datasets
from pathlib import Path

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


LR = 1e-3
IMAGE_SIZE = 64
BATCH_SIZE = 64
CHANNELS = 3
EMBEDDING_DIMS = 128
EPOCHS = 15
KLD_WEIGHT = 0.00025
WORKERS = os.cpu_count() or 0

# define the dataset path
DATASET_PATH = 'dataset/img_align_celeba/img_align_celeba'
DATASET_ATTRS_PATH = 'dataset/list_attr_celeba.csv'

# parameters for morphing process
NUM_FRAMES = 50
FPS = 5

# list of labels for image attribute modifier experiment
LABELS = ["Eyeglasses", "Smiling", "Attractive", "Male", "Blond_Hair"]


def download_celeba(root: str = DATASET_PATH):
    data_path = Path(root)
    if data_path.is_dir():
        print(f'Data folder already existed, skipping download...')
    else:
        print(f'Creating: {data_path}...')
        data_path.mkdir(parents=True, exist_ok=True)
        celeba = datasets.CelebA(
            root=root,
            download=True,
            transform=None
        )

        print('Completed downloading')


def load_model(new_model: torch.nn.Module, model_path: str) -> torch.nn.Module:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model weights not found at {model_path}")
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    new_model.load_state_dict(checkpoint['vae-celeba'])
    return new_model