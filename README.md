# DCGAN Image Generation

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) using PyTorch.
The notebook `DCGAN_celeba.ipynb` walks through the full pipeline of:

* Loading and preprocessing images
* Building the Generator and Discriminator networks
* Training a DCGAN model
* Saving generated samples during training
* Producing a final animated GIF showing training progress

## Project Overview
### 1. Importing Libraries

The project uses:

* PyTorch
* torchvision
* NumPy
* PIL
* Matplotlib
* imageio
* glob
* tqdm

### 2. Configurations & Setups

Defines hyperparameters such as:

* batch_size
* image_size
* latent_dim
* num_epochs
* learning_rate
* beta1, beta2
* output directories
* Also initializes device (CPU/GPU) and directories for saving generated images.

### 3. Dataset Loading

In this project, I used the [celeba](https://link-url-here.org) dataset for training. 

Applies transformations:

* Resize
* CenterCrop
* Normalize
* Convert to tensor
* Creates a DataLoader with shuffling and batching

### 4. Generator Architecture

A DCGAN-style generator using:

* ConvTranspose2d blocks
* BatchNorm2d
* ReLU activations
* Final Tanh output layer

### 5. Discriminator Architecture

A standard convolutional classifier using:

* Conv2d layers
* LeakyReLU
* BatchNorm2d
* Output sigmoid neuron

### 6. Model Setup

* Instantiates Generator & Discriminator
* Defines Binary Cross-Entropy Loss
* Uses Adam optimizers
* Initializes weights using DCGAN-recommended method

### 7. Training Loop

For each epoch:

* Train discriminator on:
  * real images
  * fake images
* Train generator to fool discriminator
* Save generated samples at intervals
* Record losses
* Store images for GIF creation

### 8. GIF Generation

At the end of training:

* Collects all generated sample images
* Creates an animated GIF (dcgan_final.gif)
* Saves it inside the output directory
