import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from safetensors.torch import load_file
import os

model_path = "ganyu_ff_conv.safetensors"
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

noisy_canvas = torch.rand((1, 3, 64, 64), device=device, requires_grad=True)

pixel_optimizer = torch.optim.Adam([noisy_canvas], lr=0.01)

iterations = 500

for i in range(iterations):
    pixel_optimizer.zero_grad()

    # Layer 1
    z1 = F.conv2d(noisy_canvas, w1, stride=1, padding=1)
    z1 = F.leaky_relu(z1, 0.01)
    x_hat1 = F.conv_transpose2d(z1, g1, stride=1, padding=1)
    x_hat1 = torch.sigmoid(x_hat1)
    mse1 = F.mse_loss(x_hat1, noisy_canvas)

    # Layer 2
    z2 = F.conv2d(z1, w2, stride=1, padding=1)
    z2 = F.leaky_relu(z2, 0.01)
    x_hat2 = F.conv_transpose2d(z2, g2, stride=1, padding=1)
    x_hat2 = torch.sigmoid(x_hat2)
    mse2 = F.mse_loss(x_hat2, z1)

    # Layer 3
    z3 = F.conv2d(z2, w3, stride=1, padding=1)
    z3 = F.leaky_relu(z3, 0.01)
    x_hat3 = F.conv_transpose2d(z3, g3, stride=1, padding=1)
    x_hat3 = torch.sigmoid(x_hat3)
    mse3 = F.mse_loss(x_hat3, z2)

    # Total Energy (minimize this)
    # BUT we want to MAXIMIZE activity to prevent flat colors
    activity = torch.mean(torch.abs(z3))

    loss = (mse1 + mse2 + mse3) - 0.01 * activity

    loss.backward()
    pixel_optimizer.step()

    with torch.no_grad():
        noisy_canvas.clamp_(0.0, 1.0)

    if (i+1) % 100 == 0:
        print(f"Iteration {i+1:3d}/{iterations} | MSE: {(mse1+mse2+mse3).item():.4f} | Activity: {activity.item():.4f}")

final_image = noisy_canvas.detach().cpu()
vutils.save_image(final_image, "activity_face.png")
print("Saved activity_face.png")
