import torch
import torch.nn as nn
from torchvision import transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import os
import random
from pathlib import Path

from pyimagesearch import config, network, data_utils

def load_best_model():
    """Load the best performing VAE model"""
    model = network.CelebVAE(config.CHANNELS, config.EMBEDDING_DIMS).to(config.DEVICE)
    
    # Load the best weights
    best_weights_path = 'output/model_weights/best_vae_celeba.pt'
    if not os.path.exists(best_weights_path):
        raise FileNotFoundError(f"Best model weights not found at {best_weights_path}")
    
    checkpoint = torch.load(best_weights_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['vae-celeba'])
    model.eval()
    
    print(f"✅ Loaded best model weights from {best_weights_path}")
    return model

def preprocess_image(image_path):
    """Preprocess a single image for the model"""
    transform = T.Compose([
        T.CenterCrop(148),
        T.Resize(config.IMAGE_SIZE),
        T.ToTensor()
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)# Add batch dimension
    return image

def postprocess_image(tensor):
    """Convert tensor back to displayable image"""
    # Clamp values to [-1, 1] range (due to tanh activation)
    tensor = torch.clamp(tensor, -1, 1)
    # Convert from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    # Convert to numpy and transpose for matplotlib
    image = tensor.squeeze(0).cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    return image

def reconstruct_random_images(model, num_images=8, save_results=True):
    """Reconstruct random images from the dataset"""
    
    # Get random image paths from dataset
    dataset_path = Path(config.DATASET_PATH)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {config.DATASET_PATH}")
    
    image_files = list(dataset_path.glob('*.jpg'))
    if len(image_files) == 0:
        raise FileNotFoundError(f"No .jpg files found in {config.DATASET_PATH}")
    
    # Select random images
    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    
    print(f"🔄 Reconstructing {len(selected_images)} random images...")
    
    # Create subplot
    fig, axes = plt.subplots(2, len(selected_images), figsize=(3*len(selected_images), 6))
    if len(selected_images) == 1:
        axes = axes.reshape(2, 1)
    
    with torch.no_grad():
        for i, img_path in enumerate(selected_images):
            # Load and preprocess original image
            original_tensor = preprocess_image(img_path).to(config.DEVICE)
            
            # Reconstruct image
            reconstructed_tensor, _, _, _ = model(original_tensor)
            
            # Convert to displayable format
            original_img = postprocess_image(original_tensor)
            reconstructed_img = postprocess_image(reconstructed_tensor)
            
            # Plot original image
            axes[0, i].imshow(original_img)
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Plot reconstructed image
            axes[1, i].imshow(reconstructed_img)
            axes[1, i].set_title(f'Reconstructed {i+1}')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_results:
        output_path = 'output/reconstruction_results.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 Results saved to {output_path}")
    
    # plt.show()
    return fig

def reconstruct_specific_images(model, image_paths, save_results=True):
    """Reconstruct specific images given their paths"""
    
    print(f"🔄 Reconstructing {len(image_paths)} specific images...")
    
    # Create subplot
    fig, axes = plt.subplots(2, len(image_paths), figsize=(3*len(image_paths), 6))
    if len(image_paths) == 1:
        axes = axes.reshape(2, 1)
    
    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            if not os.path.exists(img_path):
                print(f"⚠️ Warning: Image not found at {img_path}")
                continue
                
            # Load and preprocess original image
            original_tensor = preprocess_image(img_path).to(config.DEVICE)
            
            # Reconstruct image
            reconstructed_tensor, _, _, _ = model(original_tensor)
            
            # Convert to displayable format
            original_img = postprocess_image(original_tensor)
            reconstructed_img = postprocess_image(reconstructed_tensor)
            
            # Plot original image
            axes[0, i].imshow(original_img)
            axes[0, i].set_title(f'Original\n{os.path.basename(img_path)}')
            axes[0, i].axis('off')
            
            # Plot reconstructed image
            axes[1, i].imshow(reconstructed_img)
            axes[1, i].set_title(f'Reconstructed\n{os.path.basename(img_path)}')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_results:
        output_path = 'output/specific_reconstruction_results.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 Results saved to {output_path}")
    
    # plt.show()
    return fig

def generate_new_samples(model, num_samples=8, save_results=True):
    """Generate completely new images by sampling from the latent space"""
    
    print(f"🎨 Generating {num_samples} new samples from latent space...")
    
    with torch.no_grad():
        # Sample from standard normal distribution
        z = torch.randn(num_samples, config.EMBEDDING_DIMS).to(config.DEVICE)
        
        # Decode samples
        generated_images = model.decode(z)
        
        # Create subplot
        cols = 4
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            row = i // cols
            col = i % cols
            
            # Convert to displayable format
            img = postprocess_image(generated_images[i:i+1])
            
            axes[row, col].imshow(img)
            axes[row, col].set_title(f'Generated {i+1}')
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(num_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_results:
        output_path = 'output/generated_samples.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 Results saved to {output_path}")
    
    # plt.show()
    return fig

def interpolate_between_images(model, img_path1, img_path2, num_steps=10, save_results=True):
    """Interpolate between two images in the latent space"""
    
    print(f"🔄 Interpolating between two images with {num_steps} steps...")
    
    with torch.no_grad():
        # Load and preprocess both images
        img1_tensor = preprocess_image(img_path1).to(config.DEVICE)
        img2_tensor = preprocess_image(img_path2).to(config.DEVICE)
        
        # Encode both images to latent space
        mu1, log_var1 = model.encode(img1_tensor)
        mu2, log_var2 = model.encode(img2_tensor)
        
        # Use means for interpolation (ignore variance for cleaner interpolation)
        z1 = mu1
        z2 = mu2
        
        # Create interpolation steps
        interpolation_ratios = np.linspace(0, 1, num_steps)
        
        fig, axes = plt.subplots(1, num_steps, figsize=(2*num_steps, 2))
        
        for i, ratio in enumerate(interpolation_ratios):
            # Interpolate in latent space
            z_interp = (1 - ratio) * z1 + ratio * z2
            
            # Decode interpolated latent vector
            reconstructed = model.decode(z_interp)
            
            # Convert to displayable format
            img = postprocess_image(reconstructed)
            
            axes[i].imshow(img)
            axes[i].set_title(f'Step {i+1}')
            axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_results:
        output_path = 'output/interpolation_results.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 Results saved to {output_path}")
    
    # plt.show()
    return fig

def main():
    """Main function to demonstrate VAE capabilities"""
    print("🚀 Starting VAE Testing...")
    print(f"Device: {config.DEVICE}")
    
    try:
        # Load the best model
        model = load_best_model()
        
        print("\n" + "="*50)
        print("Choose an option:")
        print("1. Reconstruct random images from dataset")
        print("2. Generate new samples from latent space")
        print("3. Reconstruct specific images (you'll need to provide paths)")
        print("4. Interpolate between two images")
        print("5. Run all demonstrations")
        print("="*50)
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            reconstruct_random_images(model, num_images=6)
            
        elif choice == '2':
            generate_new_samples(model, num_samples=8)
            
        elif choice == '3':
            print("Please provide image paths (relative to dataset folder or absolute paths)")
            paths_input = input("Enter image paths separated by commas: ").strip()
            image_paths = [path.strip() for path in paths_input.split(',')]
            
            # Convert relative paths to absolute if needed
            dataset_path = Path(config.DATASET_PATH)
            absolute_paths = []
            for path in image_paths:
                if not os.path.isabs(path):
                    # Assume it's relative to dataset folder
                    absolute_path = dataset_path / path
                else:
                    absolute_path = Path(path)
                absolute_paths.append(str(absolute_path))
            
            reconstruct_specific_images(model, absolute_paths)
            
        elif choice == '4':
            dataset_path = Path(config.DATASET_PATH)
            image_files = list(dataset_path.glob('*.jpg'))[:10]  # Get first 10 for example
            
            print("Available images for interpolation:")
            for i, img_path in enumerate(image_files):
                print(f"{i}: {img_path.name}")
            
            try:
                idx1 = int(input("Enter index of first image: "))
                idx2 = int(input("Enter index of second image: "))
                interpolate_between_images(model, str(image_files[idx1]), str(image_files[idx2]))
            except (ValueError, IndexError):
                print("Invalid indices. Using first two images.")
                interpolate_between_images(model, str(image_files[0]), str(image_files[1]))
                
        elif choice == '5':
            print("🎬 Running all demonstrations...")
            reconstruct_random_images(model, num_images=4)
            generate_new_samples(model, num_samples=6)
            
            # Pick two random images for interpolation
            dataset_path = Path(config.DATASET_PATH)
            image_files = list(dataset_path.glob('*.jpg'))
            if len(image_files) >= 2:
                selected = random.sample(image_files, 2)
                interpolate_between_images(model, str(selected[0]), str(selected[1]), num_steps=8)
            
        else:
            print("Invalid choice. Running random reconstruction by default.")
            reconstruct_random_images(model, num_images=6)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
