import image_encoder

image_path = '/home/asalvi/code_workspace/VAE/imgs_bag_1/56.jpg'
encoder_model_path = '/home/asalvi/code_workspace/VAE/torch_vae/autoencoder.pth'
encoded_image = image_encoder.encode_image(image_path, encoder_model_path)


print(encoded_image.shape)
