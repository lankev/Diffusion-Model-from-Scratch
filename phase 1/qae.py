# qae.py — Autoencodeur avec goulot quantique (FAST MODE + métriques)
# Usage : python qae.py

import os, math, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms, utils as vutils
from skimage.metrics import structural_similarity as ssim
import pennylane as qml

# ===================== Config =====================
DATASET    = "FashionMNIST"   # "MNIST" ou "FashionMNIST"
BATCH      = 8                # petit pour accélérer (augmente plus tard)
EPOCHS     = 1                # 1 pour valider vite (augmente ensuite)
LATENT     = 4                # nb de qubits (4 rapide ; 8 après)
LAYERS     = 2                # profondeur du circuit variatique
LR         = 1e-3
FAST_STEPS = 30               # limite de mini-batches/epoch (0 = désactivé)
OUTDIR     = "runs"
SEED       = 42

os.makedirs(OUTDIR, exist_ok=True)

# Fixer les seeds (repeatable-ish)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device_torch = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== Data =====================
tfm = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(32),
    transforms.ToTensor(),  # [0,1]
])

if DATASET.lower() == "mnist":
    trainset = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    testset  = datasets.MNIST("./data", train=False, download=True, transform=tfm)
else:
    trainset = datasets.FashionMNIST("./data", train=True, download=True, transform=tfm)
    testset  = datasets.FashionMNIST("./data", train=False, download=True, transform=tfm)

# Windows: num_workers=0 conseillé
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True,  num_workers=0)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=BATCH, shuffle=False, num_workers=0)

# ===================== Réseaux =====================
class Encoder(nn.Module):
    def __init__(self, out_dim=LATENT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),   # 32->16
            nn.Conv2d(32,64,3, 2, 1), nn.ReLU(),    # 16->8
            nn.Conv2d(64,128,3,2,1), nn.ReLU(),     # 8->4
            nn.AdaptiveAvgPool2d(1)                 # 4->1
        )
        self.fc = nn.Linear(128, out_dim)
    def forward(self, x):
        z = self.net(x).flatten(1)
        return self.fc(z)

class Decoder(nn.Module):
    def __init__(self, in_dim=LATENT):
        super().__init__()
        self.fc = nn.Linear(in_dim, 128)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="nearest"),        # 1x1 -> 4x4
            nn.Conv2d(128,64,3,1,1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),        # 4x4 -> 8x8
            nn.Conv2d(64,32,3,1,1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),        # 8x8 -> 16x16
            nn.Conv2d(32,16,3,1,1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),        # 16x16 -> 32x32
            nn.Conv2d(16,1,3,1,1),
            nn.Sigmoid()
        )
    def forward(self, z):
        h = self.fc(z).view(-1,128,1,1)
        return self.up(h)

# ===================== Dispositif quantique =====================
try:
    qdev = qml.device("lightning.qubit", wires=LATENT)
except Exception:
    qdev = qml.device("default.qubit", wires=LATENT)

@qml.qnode(qdev, interface="torch")
def qnode(inputs, weights):
    # inputs: (LATENT,)
    for i in range(LATENT):
        qml.RY(inputs[i], wires=i)
    for l in range(LAYERS):
        for i in range(LATENT):
            a, b, c = weights[l, i]
            qml.Rot(a, b, c, wires=i)
        # intrication en anneau
        for i in range(LATENT):
            qml.CNOT(wires=[i, (i + 1) % LATENT])
    return [qml.expval(qml.PauliZ(i)) for i in range(LATENT)]

weight_shapes = {"weights": (LAYERS, LATENT, 3)}
q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

class QuantumBottle(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.pre = nn.Tanh()
        self.scale = nn.Parameter(torch.tensor(math.pi))
        self.q = q_layer
        self.post = nn.Linear(latent, latent)
    def forward(self, z):
        # z: (B, LATENT) — appel par échantillon pour éviter les soucis de batch
        z = self.pre(z) * self.scale
        outs = []
        for i in range(z.shape[0]):
            outs.append(self.q(z[i]))        # (LATENT,)
        zq = torch.stack(outs, dim=0)         # (B, LATENT)
        return self.post(zq)

class QAE(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.enc = Encoder(out_dim=latent)
        self.bottle = QuantumBottle(latent=latent)
        self.dec = Decoder(in_dim=latent)
    def forward(self, x):
        z = self.enc(x)
        zq = self.bottle(z)
        xr = self.dec(zq)
        return xr

# ===================== Métriques (PSNR/SSIM) =====================
def batch_psnr(x_rec: torch.Tensor, x_ref: torch.Tensor, data_range: float = 1.0):
    xr, xt = x_rec.detach().cpu().numpy(), x_ref.detach().cpu().numpy()
    out = []
    for i in range(xt.shape[0]):
        mse = np.mean((xr[i] - xt[i]) ** 2)
        out.append(float("inf") if mse <= 1e-12 else 20.0 * math.log10(data_range) - 10.0 * math.log10(mse))
    return float(np.mean(out)), out

def batch_ssim(x_rec: torch.Tensor, x_ref: torch.Tensor, data_range: float = 1.0):
    xr, xt = x_rec.detach().cpu().numpy(), x_ref.detach().cpu().numpy()
    out = [ssim(xt[i,0], xr[i,0], data_range=data_range, gaussian_weights=True, win_size=7) for i in range(xt.shape[0])]
    return float(np.mean(out)), out

# ===================== Entraînement =====================
model = QAE().to(device_torch)
opt = optim.Adam(model.parameters(), lr=LR)
crit = nn.MSELoss()

def eval_epoch(dl):
    model.eval()
    total, n, steps = 0.0, 0, 0
    with torch.no_grad():
        for x,_ in dl:
            x = x.to(device_torch)
            xr = model(x)
            loss = crit(xr, x)
            total += loss.item() * x.size(0)
            n += x.size(0)
            steps += 1
            if FAST_STEPS and steps >= FAST_STEPS:
                break
    return total / max(n,1)

def train_epoch(dl):
    model.train()
    total, n, steps = 0.0, 0, 0
    for x,_ in dl:
        x = x.to(device_torch)
        opt.zero_grad(set_to_none=True)
        xr = model(x)
        loss = crit(xr, x)
        loss.backward()
        opt.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
        steps += 1
        if FAST_STEPS and steps >= FAST_STEPS:
            break
    return total / max(n,1)

for e in range(1, EPOCHS+1):
    tr = train_epoch(trainloader)
    va = eval_epoch(testloader)
    print(f"[Epoch {e:02d}] train_mse={tr:.4f} | val_mse={va:.4f}")

    # Visualisation + métriques rapides
    with torch.no_grad():
        x,_ = next(iter(testloader))
        x = x.to(device_torch)[:8]
        xr = model(x).cpu()

        vutils.save_image(x.cpu(),  os.path.join(OUTDIR, f"e{e:02d}_inputs.png"), nrow=8, normalize=True)
        vutils.save_image(xr,       os.path.join(OUTDIR, f"e{e:02d}_recons.png"), nrow=8, normalize=True)

        psnr_mean, _ = batch_psnr(xr, x.cpu(), data_range=1.0)
        ssim_mean, _ = batch_ssim(xr, x.cpu(), data_range=1.0)
        print(f"   -> PSNR={psnr_mean:.2f} dB | SSIM={ssim_mean:.3f}")

# Évaluation globale courte (respecte FAST_STEPS)
with torch.no_grad():
    psnrs, ssims = [], []
    steps = 0
    for xb,_ in testloader:
        xb = xb.to(device_torch)
        xr = model(xb).cpu()
        p,_ = batch_psnr(xr, xb.cpu(), data_range=1.0)
        s,_ = batch_ssim(xr, xb.cpu(), data_range=1.0)
        psnrs.append(p); ssims.append(s)
        steps += 1
        if FAST_STEPS and steps >= FAST_STEPS:
            break
    if psnrs:
        print(f"Global ~ PSNR={sum(psnrs)/len(psnrs):.2f} dB | SSIM={sum(ssims)/len(ssims):.3f}")

print("Terminé. Les images sont dans le dossier:", OUTDIR)
