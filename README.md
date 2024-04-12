# ImageEncoder
An image encoder to reduce the latent dimensions of the image to reduce sim2real gap in Husky

# Autoencoder Documentation

This document provides an overview and documentation for the Autoencoder code provided.

## Overview

The Autoencoder is a neural network architecture used for unsupervised learning of efficient codings. It consists of an encoder network that compresses the input data into a lower-dimensional latent space and a decoder network that reconstructs the original input data from the latent space representation.

## Code Structure

The code is organized into the following sections:

1. **Imports**: Import necessary libraries and modules.
2. **ImageDataset Class**: Defines a custom dataset class for loading and preprocessing image data.
3. **Autoencoder Class**: Defines the Autoencoder model architecture using PyTorch.
4. **Hyperparameters**: Specifies the hyperparameters used for training the Autoencoder.
5. **Device Configuration**: Sets the device (CPU or GPU) for training based on availability.
6. **Dataset and DataLoader**: Loads the dataset using the custom dataset class and creates a DataLoader for batching and shuffling.
7. **Model Initialization**: Initializes the Autoencoder model, loss function, and optimizer.
8. **Training Loop**: Trains the Autoencoder model on the dataset for the specified number of epochs.
9. **Visualization**: Visualizes original and reconstructed images after training.
10. **Model Saving**: Saves the trained encoder and decoder models to disk.

## Customization

- **Changing Latent Dimension**: The latent dimension of the Autoencoder can be modified by changing the `latent_dim` variable in the hyperparameters section.

## Usage

To use the Autoencoder code:

1. Ensure all necessary libraries and modules are installed.
2. Update the dataset path in the `ImageDataset` class instantiation.
3. Customize hyperparameters as needed.
4. Run the code to train the Autoencoder.
5. Visualize original and reconstructed images to evaluate performance.
6. Save the trained encoder and decoder models for future use.

## Dependencies

- PyTorch
- torchvision
- NumPy
- Matplotlib
- OpenCV (cv2)
- PIL (Python Imaging Library)

## License

This code is provided under the [MIT License](https://opensource.org/licenses/MIT).

