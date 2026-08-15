import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from safetensors.torch import load_file
import os

model_path = "ganyu_ff_conv.safetensors"
if not os.path.exists(model_path):
    from huggingface_hub import hf_hub_download
    repo_id = "RinKana/RGL-AE-AL-186K"
    filename = "anime_rgl_ae.safetensors"
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    os.symlink(path, "ganyu_ff_conv.safetensors")

state_dict = load_file(model_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

w1 = state_dict["layer1.W"].to(device)
g1 = state_dict["layer1.G"].to(device)
w2 = state_dict["layer2.W"].to(device)
g2 = state_dict["layer2.G"].to(device)
w3 = state_dict["layer3.W"].to(device)
g3 = state_dict["layer3.G"].to(device)

for w in [w1, g1, w2, g2, w3, g3]:
    w.requires_grad = False

# Initialize random noise canvas
noisy_canvas = torch.rand((1, 3, 64, 64), device=device, requires_grad=True)

iterations = 2000
lr = 100.0  # learning rate / step size for Langevin
noise_scale = 0.05

for i in range(iterations):
    if noisy_canvas.grad is not None:
        noisy_canvas.grad.zero_()

    z1 = F.conv2d(noisy_canvas, w1, stride=1, padding=1)
    z1 = F.leaky_relu(z1, 0.01)
    x_hat1 = F.conv_transpose2d(z1, g1, stride=1, padding=1)
    x_hat1 = torch.sigmoid(x_hat1)
    mse1 = F.mse_loss(x_hat1, noisy_canvas)

    z2 = F.conv2d(z1, w2, stride=1, padding=1)
    z2 = F.leaky_relu(z2, 0.01)
    x_hat2 = F.conv_transpose2d(z2, g2, stride=1, padding=1)
    x_hat2 = torch.sigmoid(x_hat2)
    mse2 = F.mse_loss(x_hat2, z1)

    z3 = F.conv2d(z2, w3, stride=1, padding=1)
    z3 = F.leaky_relu(z3, 0.01)
    x_hat3 = F.conv_transpose2d(z3, g3, stride=1, padding=1)
    x_hat3 = torch.sigmoid(x_hat3)
    mse3 = F.mse_loss(x_hat3, z2)

    # Total Energy (we want to minimize this)
    loss = mse1 + mse2 + mse3
    loss.backward()

    # Langevin step
    with torch.no_grad():
        noisy_canvas.data = noisy_canvas.data - (lr / 2) * noisy_canvas.grad
        if i < iterations - 500:  # cool down noise towards the end
            noisy_canvas.data = noisy_canvas.data + noise_scale * torch.randn_like(noisy_canvas)
        noisy_canvas.data.clamp_(0.0, 1.0)

    if (i+1) % 100 == 0:
        print(f"Iteration {i+1:3d}/{iterations} | Energy: {loss.item():.4f}")

final_image = noisy_canvas.detach().cpu()
vutils.save_image(final_image, "langevin_face.png")
print("Saved langevin_face.png")
