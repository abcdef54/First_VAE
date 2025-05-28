# CelebA Variational Autoencoder (VAE)

A PyTorch implementation of a Variational Autoencoder (VAE) trained on the CelebA dataset for generating and reconstructing celebrity face images.

## 🎯 Overview

This project implements a convolutional VAE that can:
- **Reconstruct** celebrity face images with high fidelity
- **Generate** new synthetic celebrity faces by sampling from the latent space
- **Interpolate** between different faces in the latent space
- **Encode** images into a meaningful 128-dimensional latent representation

## 🏗️ Architecture

The VAE consists of two main components:

### Encoder
- 5 convolutional blocks with increasing channels: `[32, 64, 128, 256, 512]`
- Each block: `Conv2d → BatchNorm2d → LeakyReLU`
- Fully connected layers to produce mean (μ) and log-variance (σ²) for the latent space

### Decoder
- Linear layer to expand latent vector back to feature maps
- 4 transpose convolutional blocks with decreasing channels
- Final layer with `Tanh` activation for pixel values in [-1, 1] range

### Key Features
- **Latent Dimension**: 128
- **Image Size**: 64×64 RGB
- **Reparameterization Trick**: Enables differentiable sampling from the latent distribution
- **Xavier Weight Initialization**: For stable training

## 📁 Project Structure

```
VAE/
├── pyimagesearch/
│   ├── __init__.py
│   ├── config.py          # Configuration parameters
│   ├── data_utils.py      # Dataset handling utilities
│   ├── model_utils.py     # Loss functions and utilities
│   └── network.py         # VAE architecture
├── dataset/               # CelebA dataset directory
├── output/
│   ├── model_weights/     # Saved model checkpoints
│   └── training_progress/ # Training visualizations
├── train.py              # Training script
├── test.py               # Testing and inference script
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision matplotlib numpy pillow tqdm
```

### Dataset Setup

The model is trained on the CelebA dataset. The dataset will be automatically downloaded when you first run the training script.

### Training

```bash
python train.py
```

**Training Features:**
- **Data Augmentation**: Random horizontal flips and center cropping
- **Validation Split**: 10% of the dataset
- **Learning Rate Scheduling**: Exponential decay (γ=0.95)
- **Gradient Clipping**: Prevents exploding gradients
- **Best Model Saving**: Automatically saves the best performing model

**Training Configuration:**
- Learning Rate: 1e-3
- Batch Size: 64
- Epochs: 15
- KL Divergence Weight: 0.00025
- Optimizer: Adam

### Testing and Inference

```bash
python test.py
```

The test script provides several options:
1. **Reconstruct Random Images**: Test reconstruction quality on random dataset samples
2. **Generate New Samples**: Create entirely new faces by sampling from the latent space
3. **Reconstruct Specific Images**: Test on your own image paths
4. **Interpolate Between Images**: Smooth transitions between two faces in latent space
5. **Run All Demonstrations**: Execute all the above tests

## 📊 Loss Function

The VAE uses a composite loss function:

```
Total Loss = Reconstruction Loss + β × KL Divergence Loss
```

- **Reconstruction Loss**: Measures how well the decoded image matches the original
- **KL Divergence Loss**: Regularizes the latent space to follow a standard normal distribution
- **β (KLD_WEIGHT)**: 0.00025 - balances reconstruction quality vs. latent space regularity

## 🎨 Results

The trained model can:
- Reconstruct celebrity faces with high visual fidelity
- Generate diverse, realistic new celebrity faces
- Perform smooth interpolations between different faces
- Learn meaningful latent representations that capture facial features

Results are automatically saved to the `output/` directory:
- `reconstruction_results.png`: Original vs. reconstructed images
- `generated_samples.png`: Newly generated faces
- `interpolation_results.png`: Face interpolation sequences

## ⚙️ Configuration

Key parameters in `pyimagesearch/config.py`:

```python
IMAGE_SIZE = 64          # Input image resolution
EMBEDDING_DIMS = 128     # Latent space dimensionality
BATCH_SIZE = 64          # Training batch size
EPOCHS = 15              # Training epochs
KLD_WEIGHT = 0.00025     # KL divergence loss weight
LR = 1e-3                # Learning rate
```

## 🔧 Model Utilities

- **Automatic Model Loading**: Best weights are loaded automatically during inference
- **Device Detection**: Automatically uses GPU if available
- **Progress Tracking**: Training progress with tqdm progress bars
- **Flexible Image Processing**: Handles various image formats and sizes

## 🎯 Use Cases

- **Face Generation**: Create synthetic celebrity faces for creative projects
- **Image Compression**: Encode images into compact 128-dimensional representations
- **Face Morphing**: Create smooth transitions between different faces
- **Feature Analysis**: Explore the latent space to understand facial feature representations
- **Data Augmentation**: Generate additional training data for other face-related tasks

## 📈 Performance Tips

- **GPU Training**: Significantly faster training with CUDA-compatible GPU
- **Batch Size**: Adjust based on available GPU memory
- **KL Weight**: Lower values prioritize reconstruction, higher values improve generation
- **Training Time**: Approximately 2-3 hours on modern GPU for 15 epochs

## 🤝 Contributing

Feel free to submit issues, suggestions, or improvements. This implementation serves as a solid foundation for experimenting with VAE architectures and face generation tasks.

## 📚 References

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - Original VAE paper
- [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) - Large-scale CelebFaces Attributes Dataset

---

*Happy face generating! 🎭*