import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

from pyimagesearch import config, model_utils, data_utils, network
import os
from tqdm.auto import tqdm

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# create the training_progress directory inside the output directory
training_progress_dir = os.path.join(output_dir, 'training_progress')
os.makedirs(training_progress_dir, exist_ok=True)

# create the model_weights directory inside the output directory for storing autoencoder weights
model_weights_dir = os.path.join(output_dir, 'model_weights')
os.makedirs(model_weights_dir, exist_ok=True)

# define model_weights path including best weights
'''This path points to where the best weights of the model (determined by validation loss) will be saved. The file is named “best_vae_celeba.pt”.'''
MODEL_BEST_WEIGHT_PATH = os.path.join(model_weights_dir, 'best_vae_celeba.pt')

'''This path points to where the final weights of the model will be saved, named “vae_celeba.pt”.'''
MODEL_WEIGHT_PATH = os.path.join(model_weights_dir, 'vae_celeba.pt')

# Define the transformations
train_transforms = T.Compose([
    T.RandomHorizontalFlip(),   # A common data augmentation technique to increase the diversity of the training data and make the model more robust.
    T.CenterCrop(148),  # crops the given image at the center to have a size of 148x148 pixels.
    T.Resize(config.IMAGE_SIZE),
    T.ToTensor()
])

val_transforms = T.Compose([
    T.RandomHorizontalFlip(),
    T.CenterCrop(148),
    T.Resize(config.IMAGE_SIZE),
    T.ToTensor()
])

# Instantiate the dataset
if not os.path.exists(config.DATASET_PATH):
    config.download_celeba(config.DATASET_PATH)

celeba_dataset = data_utils.CelebADataset(config.DATASET_PATH, transform=train_transforms)

# Define the size of the validation set
val_size = int(len(celeba_dataset)*0.1) # 10%
train_size = len(celeba_dataset) - val_size

train_dataset, val_dataset = random_split(celeba_dataset, [train_size, val_size])

# Define the dataloaders
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.WORKERS,
    pin_memory=True
)
'''
The pin_memory=True argument is used to speed up data transfer between the CPU and GPU.
When set to True, it will allocate the Tensors in CUDA pinned memory, which enables faster data transfer to the CUDA device.
'''
val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.WORKERS,
    pin_memory=True
)


model = network.CelebVAE(config.CHANNELS, config.EMBEDDING_DIMS).to(config.DEVICE)
model = config.load_model(model, 'output/model_weights/best_vae_celeba.pt')

# Instantiate optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=config.LR)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
'''
The chosen scheduler is ExponentialLR, which multiplies the learning rate by a factor (gamma) at each epoch.
In this case, the learning rate is multiplied by 0.95 at the end of each epoch, leading to a gradual decrease in the learning rate as training progresses.
This approach can help in stabilizing training and achieving better convergence.
'''

# initialize the best validation loss as infinity
best_val_loss = float('inf')

print('Training started')

def validate(model: nn.Module, val_dataloader, device):
    model.eval()
    with torch.inference_mode():
        total_length = 0
        total_loss = 0.0
        for x in val_dataloader:
            x = x.to(device)

            prediction = model(x)
            loss = model_utils.VAE_Loss(prediction, config.KLD_WEIGHT)
            total_loss += loss['loss'].item()
            total_length += x.size(0)
        
    return total_loss / total_length
        

for epoch in tqdm(range(config.EPOCHS)):
    running_loss = 0.0
    for i, x in enumerate(train_dataloader):
        x = x.to(config.DEVICE)

        optimizer.zero_grad()
        predictions = model(x)
        total_loss = model_utils.VAE_Loss(predictions, config.KLD_WEIGHT)

        total_loss['loss'].backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        running_loss += total_loss['loss'].item()

    # compute average loss for the epoch
    train_loss = running_loss / len(train_dataloader)
    # compute validation loss for the epoch
    val_loss = validate(model, val_dataloader, config.DEVICE)

    if val_loss <= best_val_loss:
        best_val_loss = val_loss
        torch.save(
            {"vae-celeba": model.state_dict()},
            MODEL_BEST_WEIGHT_PATH,
        )
    
    # Save current epoch weights
    torch.save(
        {"vae-celeba": model.state_dict()},
        MODEL_WEIGHT_PATH,
    )
    print(
        f"Epoch {epoch+1}/{config.EPOCHS}, Batch {i+1}/{len(train_dataloader)}, "
        f"Total Loss: {total_loss['loss'].detach().item():.4f}, "
        f"Reconstruction Loss: {total_loss['Reconstruction_loss']:.4f}, "
        f"KL Divergence Loss: {total_loss['KLD']:.4f}",
        f"Val Loss: {val_loss:.4f}",
    )
    scheduler.step()



