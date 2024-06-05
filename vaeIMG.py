import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# Define a custom dataset class
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10
latent_dim = 32  # Specify latent dimension

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a dataset and dataloader
transform = transforms.Compose([
    transforms.Resize((96, 320)),
    transforms.ToTensor()
])

dataset = ImageDataset(root_dir='/home/asalvi/code_workspace/tmp/image_data/AEImg/', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
model = Autoencoder(latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
total_steps = len(dataloader)
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        inputs = data.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), '/home/asalvi/code_workspace/VAE/torch_vae/autoencoder.pth')

print('Training finished.')

# Visualization
# Select some images from the dataset for visualization
original_images = next(iter(dataloader))

# Pass the original images through the model to get reconstructed images
reconstructed_images = model(original_images.to(device)).cpu().detach()

# Function to display images
def show_images(original, reconstructed, num_images=5):
    fig, axs = plt.subplots(2, num_images, figsize=(15, 7))
    for i in range(num_images):
        axs[0, i].imshow(np.transpose(original[i], (1, 2, 0)))  # Original image
        axs[0, i].set_title('Original')
        axs[0, i].axis('off')
        axs[1, i].imshow(np.transpose(reconstructed[i], (1, 2, 0)))  # Reconstructed image
        axs[1, i].set_title('Reconstructed')
        axs[1, i].axis('off')
    plt.show()

# Visualize the original and reconstructed images
show_images(original_images, reconstructed_images)
