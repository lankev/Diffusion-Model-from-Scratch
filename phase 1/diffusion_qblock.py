# diffusion_qblock.py — DDPM light + U-Net avec bloc quantique au bottleneck
# Usage : python diffusion_qblock.py
# Dépendances : torch torchvision pennylane

import os, math, random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, utils as vutils
import pennylane as qml

# ===================== Config =====================
DATASET       = "FashionMNIST"   # ou "MNIST"
IMG_SIZE      = 32
CHANNELS      = 1

BATCH         = 16
EPOCHS        = 5
FAST_STEPS    = 0
LR            = 2e-4

TIMESTEPS     = 200

LATENT        = 4
QLAYERS       = 1
QUANTUM_REAL  = False            # True = QNode réel (lent CPU), False = rapide

SAMPLE_STRIDE = 1

OUTDIR        = "runs_diffq"
SEED          = 123

# ===================== Device =====================
os.makedirs(OUTDIR, exist_ok=True)
random.seed(SEED)
torch.manual_seed(SEED)
torch.set_default_dtype(torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
else:
    torch.set_num_threads(8)

# ===================== Noise schedule =====================
def get_noise_schedule(T, beta_start=1e-4, beta_end=0.02, device="cpu"):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alphas_cum = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cum

BETAS, ALPHAS, ALPHAS_CUM = get_noise_schedule(TIMESTEPS, device=device)

# ===================== Data =====================
tfm = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2.0 - 1.0),
])

if DATASET.lower() == "mnist":
    trainset = datasets.MNIST("./data", train=True, download=True, transform=tfm)
else:
    trainset = datasets.FashionMNIST("./data", train=True, download=True, transform=tfm)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH,
    shuffle=True,
    num_workers=0,
    drop_last=True,
    pin_memory=(device.type == "cuda")
)

# ===================== Time embedding (CORRIGÉ) =====================
class TimeEmbedding(nn.Module):
    """
    Sinusoidal embedding -> MLP
    dim doit être pair idéalement. Output = dim*4
    """
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.lin1 = nn.Linear(dim, dim * 4)
        self.lin2 = nn.Linear(dim * 4, dim * 4)

    def forward(self, t):
        # t : [B] float
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None] * freqs[None, :]                  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]
        emb = self.lin2(F.silu(self.lin1(emb)))            # [B, dim*4]
        return emb

# ===================== Quantum bottleneck =====================
try:
    qdev = qml.device("lightning.qubit", wires=LATENT)
except Exception:
    qdev = qml.device("default.qubit", wires=LATENT)

@qml.qnode(qdev, interface="torch")
def qnode(inputs, weights):
    for i in range(LATENT):
        qml.RY(inputs[i], wires=i)
    for l in range(QLAYERS):
        for i in range(LATENT):
            a, b, c = weights[l, i]
            qml.Rot(a, b, c, wires=i)
        for i in range(LATENT):
            qml.CNOT(wires=[i, (i + 1) % LATENT])
    return [qml.expval(qml.PauliZ(i)) for i in range(LATENT)]

weight_shapes = {"weights": (QLAYERS, LATENT, 3)}
q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

class QuantumBottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.to_latent = nn.Linear(channels, LATENT)
        self.from_latent = nn.Linear(LATENT, channels)
        self.act = nn.SiLU()
        self.use_quantum = QUANTUM_REAL

    def forward(self, x):
        B, C, _, _ = x.shape
        v = self.gap(x).view(B, C)
        v = torch.tanh(self.to_latent(v)) * math.pi

        if self.use_quantum:
            v_cpu = v.detach().cpu()
            qv = torch.stack([q_layer(v_cpu[i]) for i in range(B)]).to(x.device)
        else:
            qv = torch.tanh(v)

        qv = self.act(self.from_latent(qv)).view(B, C, 1, 1)
        return x + qv

# ===================== U-Net =====================
def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.SiLU(),
    )

class UNetLite(nn.Module):
    def __init__(self, base=32, tdim=128):
        super().__init__()
        self.time_mlp = TimeEmbedding(tdim)  # output 4*tdim = 512

        self.in_conv = nn.Conv2d(1, base, 3, padding=1)
        self.down1 = conv_block(base, base)
        self.pool = nn.AvgPool2d(2)                 # 32->16
        self.down2 = conv_block(base * 2, base * 2) # in 64 -> out 64

        self.mid = conv_block(base * 2, base * 2)
        self.qb = QuantumBottleneck(base * 2)

        self.up1 = conv_block(base * 3, base)       # 64+32=96 -> 32
        self.up2 = conv_block(base * 2, base)       # 32+32=64 -> 32
        self.out = nn.Conv2d(base, 1, 1)

        self.t1 = nn.Linear(tdim * 4, base)
        self.t2 = nn.Linear(tdim * 4, base * 2)

    def forward(self, x, t):
        temb = self.time_mlp(t)  # [B,512]

        x0 = self.in_conv(x)  # [B,32,32,32]
        d1 = self.down1(x0 + self.t1(temb).view(-1, x0.size(1), 1, 1))
        p1 = self.pool(d1)    # [B,32,16,16]

        d2_in = torch.cat([p1, p1], dim=1)  # [B,64,16,16]
        d2 = self.down2(d2_in)
        d2 = d2 + self.t2(temb).view(-1, d2.size(1), 1, 1)

        m = self.qb(self.mid(d2))           # [B,64,16,16]

        u1 = F.interpolate(m, scale_factor=2, mode="nearest")   # [B,64,32,32]
        u1 = self.up1(torch.cat([u1, d1], dim=1))               # [B,96,32,32] -> [B,32,32,32]
        u2 = self.up2(torch.cat([u1, x0], dim=1))               # [B,64,32,32] -> [B,32,32,32]

        return self.out(u2)

# ===================== DDPM =====================
def sample_timesteps(n):
    return torch.randint(0, TIMESTEPS, (n,), device=device)

def add_noise(x0, t):
    noise = torch.randn_like(x0)
    sqrt_ac = torch.sqrt(ALPHAS_CUM[t])[:, None, None, None]
    sqrt_om = torch.sqrt(1.0 - ALPHAS_CUM[t])[:, None, None, None]
    return sqrt_ac * x0 + sqrt_om * noise, noise

# ===================== Training =====================
model = UNetLite().to(device)
opt = optim.AdamW(model.parameters(), lr=LR)

def train_epoch():
    model.train()
    total = 0.0
    steps = 0
    for i, (x, _) in enumerate(trainloader):
        if FAST_STEPS and i >= FAST_STEPS:
            break
        x = x.to(device, non_blocking=True)

        t = sample_timesteps(x.size(0))
        xt, eps = add_noise(x, t)
        pred = model(xt, t.float())
        loss = F.mse_loss(pred, eps)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total += loss.item()
        steps += 1
    return total / max(steps, 1)

@torch.no_grad()
def sample_images(n=16):
    model.eval()
    x = torch.randn(n, 1, IMG_SIZE, IMG_SIZE, device=device)
    stride = max(1, int(SAMPLE_STRIDE))

    # loop complet t = T-1 ... 0
    for t in range(TIMESTEPS - 1, -1, -stride):
        tt = torch.full((n,), t, device=device, dtype=torch.float32)
        eps = model(x, tt)

        beta = BETAS[t]
        alpha = ALPHAS[t]
        ac = ALPHAS_CUM[t]

        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x = (x - beta / torch.sqrt(1.0 - ac) * eps) / torch.sqrt(alpha) + torch.sqrt(beta) * noise

    x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
    return x

# ===================== Main =====================
for e in range(1, EPOCHS + 1):
    loss = train_epoch()
    print(f"[Epoch {e:02d}] loss={loss:.4f}")

    imgs = sample_images(n=16).cpu()
    vutils.save_image(imgs, f"{OUTDIR}/e{e:02d}_samples.png", nrow=8)

print("Terminé. Images dans :", OUTDIR)
