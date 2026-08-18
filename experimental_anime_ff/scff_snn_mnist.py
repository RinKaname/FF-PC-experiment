import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Grayscale, ToTensor, Normalize, Lambda
import snntorch as snn
import matplotlib.pyplot as plt
from tqdm import tqdm
from safetensors.torch import save_model

# Setup device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Dataset transforms
transform = Compose([
    Grayscale(),
    ToTensor(),
    Normalize((0,), (1,)),
    Lambda(lambda x: torch.flatten(x))
])

# Load datasets
mnist_train = MNIST('./data/', train=True, download=True, transform=transform)
train_loader = DataLoader(mnist_train, batch_size=128, shuffle=True)

mnist_test = MNIST('./data/', train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=128, shuffle=False)

# --- SCFF POS/NEG GENERATOR ---
def get_pos_neg(x):
    # SCFF positive: W(x_k + x_k) -> we just pass 2 * x_k
    x_pos = 2.0 * x
    # SCFF negative: W(x_k + x_n) -> we shift the batch
    rnd = torch.randperm(x.size(0))
    x_neg = x + x[rnd]
    return x_pos, x_neg

# --- NETWORK ARCHITECTURE ---
class LeakyLayer(nn.Linear):
    def __init__(self, in_features, out_features, activation, bias=False):
        super().__init__(in_features, out_features, bias=bias)

        if activation == "lif":
            self.activation = snn.Leaky(beta=0.8)
            self.lif = True
        else:
            self.activation = nn.ReLU()
            self.lif = False

        self.opt = AdamW(self.parameters(), lr=0.005)

        # SCFF thresholds
        self.threshold_pos = 2.0
        self.threshold_neg = 2.0
        self.lamda = 0.01 # Frobenius norm penalty

        # Learn from each batch once instead of memorizing a single batch.
        self.num_epochs = 1

    def forward(self, x):
        if self.lif == True:
            mem = self.activation.init_leaky()

        x_direction = x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-4)
        weighted_input = torch.mm(x_direction, self.weight.T.to(device))

        if self.lif == True:
            spk, potential = self.activation(weighted_input, mem)
        else:
            potential = self.activation(weighted_input)

        return potential

    def train_layer(self, x_pos, x_neg):
        tot_loss = []
        for _ in range(self.num_epochs):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)

            # SCFF Logarithmic Softplus Loss
            loss_pos = torch.log(1 + torch.exp(-g_pos + self.threshold_pos)).mean()
            loss_neg = torch.log(1 + torch.exp(g_neg - self.threshold_neg)).mean()
            penalty = self.lamda * torch.norm(g_pos).mean()
            loss = loss_pos + loss_neg + penalty

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # Prevents OOM memory crash over 60k images.
            tot_loss.append(loss.detach().item())

        output = self.forward(x_pos).detach(), self.forward(x_neg).detach()
        return (output, tot_loss)

class Net(nn.Module):
    def __init__(self, dims, activation):
        super().__init__()
        self.layers = nn.ModuleList([
            LeakyLayer(dims[d], dims[d + 1], activation) for d in range(len(dims) - 1)
        ])

    def forward_features(self, x):
        # Extract features by passing through the frozen network
        h = x
        features = []
        for layer in self.layers:
            h = layer(h)
            features.append(h)
        # We can use the last layer's feature or concatenate them all. Let's use the last one for simplicity.
        return features[-1]

    def train_network(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        layer_losses = []
        for i, layer in enumerate(self.layers):
            outputs, loss = layer.train_layer(h_pos, h_neg)
            h_pos, h_neg = outputs
            layer_losses.append(loss)
        return layer_losses

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Initialize the network
    torch.manual_seed(123)

    # The 'dims' array defines the network structure:
    # 784 = Input pixels (28x28 image)
    # 2000 = Layer 1 output neurons
    # 2000 = Layer 2 output neurons
    net = Net([784, 2000, 2000], "lif").to(device)

    total_dataset_epochs = 10

    # Lists to store the loss history globally
    history_layer_1 = []
    history_layer_2 = []

    print(f"Starting SCFF unsupervised training for {total_dataset_epochs} epochs...")
    for epoch in range(total_dataset_epochs):
        # Wrap the loader in tqdm for estimated time tracking
        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_dataset_epochs}"):
            x = x.to(device)

            x_pos, x_neg = get_pos_neg(x)

            # Extract the loss numbers before the loop moves on
            batch_losses = net.train_network(x_pos, x_neg)

            history_layer_1.append(batch_losses[0][0])
            history_layer_2.append(batch_losses[1][0])

    print("SCFF Training complete.")

    print("Freezing SCFF layers and training Linear Classifier for Evaluation...")
    # Freeze SCFF network
    for param in net.parameters():
        param.requires_grad = False

    # Create linear readout (input 2000 -> output 10)
    classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(2000, 10)
    ).to(device)

    # We train the classifier using standard CrossEntropy and Adam
    opt_cls = Adam(classifier.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    cls_epochs = 10
    for epoch in range(cls_epochs):
        classifier.train()
        for x, y in tqdm(train_loader, desc=f"Classifier Epoch {epoch+1}/{cls_epochs}"):
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                features = net.forward_features(x)

            opt_cls.zero_grad()
            logits = classifier(features)
            loss = criterion(logits, y)
            loss.backward()
            opt_cls.step()

    print("Classifier training complete. Running full evaluation...")

    # Test on the full test dataset
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_te, y_te in tqdm(test_loader, desc="Testing"):
            x_te, y_te = x_te.to(device), y_te.to(device)
            features = net.forward_features(x_te)
            logits = classifier(features)
            predicted = logits.argmax(1)
            correct += predicted.eq(y_te).sum().item()
            total += y_te.size(0)

    print('Full Test error:', 100 * (1.0 - (correct / total)), '%')

    print("Saving SCFF model weights using safetensors...")
    save_model(net, "scff_snn.safetensors")

    # Plot the saved data
    plt.plot(history_layer_1, label="Layer 1 Loss")
    plt.plot(history_layer_2, label="Layer 2 Loss")
    plt.xlabel("Training Steps (Batches)")
    plt.ylabel("SCFF Loss")
    plt.legend()
    plt.title("SCFF Spiking Neural Network Loss")
    plt.savefig("scff_loss.png")
    print("Saved plot to scff_loss.png")
