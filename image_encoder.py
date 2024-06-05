import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Define the encoder model
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


def preprocess_image(image_path):
    # Open and preprocess the image
    image = Image.open(image_path)
    image = image.crop((0, 288, 640, 480))  # Crop image
    image = image.resize((320, 96))  # Resize image
    transform = transforms.ToTensor()
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

def encode_image(image_path, encoder_model_path):
    # Load encoder model
    encoder_model = Encoder()
    encoder_model.load_state_dict(torch.load(encoder_model_path), strict=False)
    encoder_model.eval()

    # Preprocess image
    input_image = preprocess_image(image_path)

    # Encode image
    with torch.no_grad():
        encoded_image = encoder_model(input_image)
    
    return encoded_image
