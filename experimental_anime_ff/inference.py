import os
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# =====================================================================
# CONVOLUTIONAL FORWARD-FORWARD LAYER (Inference Only)
# =====================================================================
class CUDAConvFFLayer:
    def __init__(self, in_channels, out_channels, kernel_size=3):
        self.in_c = in_channels
        self.out_c = out_channels
        self.ks = kernel_size

        # Encoder weights (Conv)
        limit_w = np.sqrt(6 / (in_channels * kernel_size * kernel_size + out_channels))
        self.W = torch.from_numpy(np.random.uniform(-limit_w, limit_w, (out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32))

        # Decoder weights (Transposed Conv)
        self.G = torch.from_numpy(np.random.uniform(-(limit_w * 2.0), (limit_w * 2.0), (out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32))

        # Check for GPU and move
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.W = self.W.to(self.device)
        self.G = self.G.to(self.device)

    def forward_encoder(self, x):
        """ cuDNN Forward Pass (Autograd OFF) """
        x = x.to(self.device)
        with torch.no_grad():
            y = F.conv2d(x, self.W, stride=1, padding=1)
            y = F.leaky_relu(y, 0.01)
        return y

    def forward_decoder(self, z):
        """ cuDNN Transposed Forward Pass (Autograd OFF) """
        z = z.to(self.device)
        with torch.no_grad():
            x_pred = F.conv_transpose2d(z, self.G, stride=1, padding=1)
            x_pred = torch.clamp(x_pred, 0.0, 1.0)
        return x_pred


# =====================================================================
# GENERATOR ORCHESTRATOR FOR INFERENCE
# =====================================================================
class AnimeForwardForwardGenerator:
    def __init__(self):
        # 64x64 RGB input -> spatial compression through channels
        self.enc_layer1 = CUDAConvFFLayer(in_channels=3, out_channels=32, kernel_size=3)
        self.enc_layer2 = CUDAConvFFLayer(in_channels=32, out_channels=64, kernel_size=3)
        self.enc_layer3 = CUDAConvFFLayer(in_channels=64, out_channels=128, kernel_size=3)

    def load_safetensors(self, filepath):
        """ Load weights from Safetensors """
        tensors = load_file(filepath)
        self.enc_layer1.W = tensors["layer1.W"].to(self.enc_layer1.device)
        self.enc_layer1.G = tensors["layer1.G"].to(self.enc_layer1.device)
        self.enc_layer2.W = tensors["layer2.W"].to(self.enc_layer2.device)
        self.enc_layer2.G = tensors["layer2.G"].to(self.enc_layer2.device)
        self.enc_layer3.W = tensors["layer3.W"].to(self.enc_layer3.device)
        self.enc_layer3.G = tensors["layer3.G"].to(self.enc_layer3.device)
        print(f"Model weights successfully loaded from {filepath}.")

    def encode(self, x):
        """ Full Bottom-Up pass """
        z1 = self.enc_layer1.forward_encoder(x)
        z2 = self.enc_layer2.forward_encoder(z1)
        z3 = self.enc_layer3.forward_encoder(z2)
        return z3

    def decode(self, z):
        """ Full Top-Down pass """
        z2_pred = self.enc_layer3.forward_decoder(z)
        z1_pred = self.enc_layer2.forward_decoder(z2_pred)
        x_pred  = self.enc_layer1.forward_decoder(z1_pred)
        return x_pred

    def generate_from_noise(self, batch_size=16, spatial_size=64):
        """ Generate hallucinated faces by feeding random noise to the top-down decoder """
        # Latent space is 128 channels at the spatial size (since stride=1 padding=1 keeps size same)
        noise_th = torch.randn(batch_size, 128, spatial_size, spatial_size)
        # We can also try activating the noise to simulate the latent space properties
        noise_th = F.leaky_relu(noise_th, 0.01)

        generated_faces = self.decode(noise_th)
        return generated_faces.cpu()

if __name__ == "__main__":
    print("Downloading weights from Hugging Face Hub: RinKana/RGL-AE-186K-ganyu-face-generator")
    try:
        # Download the model from the HF hub
        model_path = hf_hub_download(repo_id="RinKana/RGL-AE-186K-ganyu-face-generator", filename="ganyu_ff_conv.safetensors")
        print(f"Downloaded weights to {model_path}")
    except Exception as e:
        print(f"Failed to download weights from HF. Ensure the repo name and filename are correct.")
        print(f"Error: {e}")
        print("Falling back to local 'ganyu_ff_conv.safetensors' if it exists...")
        model_path = "ganyu_ff_conv.safetensors"

    model = AnimeForwardForwardGenerator()

    if os.path.exists(model_path):
        model.load_safetensors(model_path)
    else:
        print(f"Weights file not found at {model_path}. Please check your HF repository or local path.")
        exit(1)

    print("\nGenerating faces from random latent noise...")

    # Generate a batch of 16 faces
    generated_images = model.generate_from_noise(batch_size=16, spatial_size=64)

    # Ensure they are clamped and ready for saving
    generated_images = torch.clamp(generated_images, 0.0, 1.0)

    # Save the output
    output_filename = "generated_faces_inference.png"
    vutils.save_image(generated_images, output_filename, nrow=4, normalize=False)
    print(f"Successfully generated and saved faces to: {output_filename}")
