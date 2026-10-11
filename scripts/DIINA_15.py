# DIINA-15 (Research Prototype)
# Fast/Slow Hybrid SSM with Adaptive Routing
# Designed for experimentation on LRA-style tasks

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Selective State Space Module
# -----------------------------
class SelectiveSSM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.A = nn.Parameter(torch.randn(dim) * 0.02)
        self.B = nn.Linear(dim, dim)
        self.C = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        h = torch.zeros(B, D, device=x.device)
        outputs = []

        for t in range(T):
            u = x[:, t]
            h = torch.tanh(self.A * h + self.B(u))
            y = self.C(h)
            outputs.append(y)

        return torch.stack(outputs, dim=1)


# -----------------------------
# Fast Path (local mixing)
# -----------------------------
class FastPath(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x [B,T,D]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return self.proj(x)


# -----------------------------
# Slow Path (SSM memory)
# -----------------------------
class SlowPath(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ssm = SelectiveSSM(dim)

    def forward(self, x):
        return self.ssm(x)


# -----------------------------
# Adaptive Router
# -----------------------------
class AdaptiveRouter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.router = nn.Linear(dim, 2)

    def forward(self, x):
        g = torch.softmax(self.router(x), dim=-1)
        return g


# -----------------------------
# DIINA Block
# -----------------------------
class DIINABlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.fast = FastPath(dim)
        self.slow = SlowPath(dim)
        self.router = AdaptiveRouter(dim)

        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        residual = x

        fast_out = self.fast(x)
        slow_out = self.slow(x)

        gates = self.router(x)

        mixed = gates[..., 0:1] * fast_out + gates[..., 1:2] * slow_out

        x = self.norm(residual + mixed)
        x = x + self.ff(x)

        return x


# -----------------------------
# DIINA-15 Model
# -----------------------------
class DIINA15(nn.Module):
    def __init__(self, vocab_size=256, dim=128, depth=4, seq_len=64):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            DIINABlock(dim) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.head(x)


# -----------------------------
# Simple training demo
# -----------------------------

def run_debug():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    model = DIINA15().to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    B = 8
    T = 64
    V = 256

    for step in range(40):

        x = torch.randint(0, V, (B, T)).to(device)
        y = torch.roll(x, shifts=-1, dims=1)

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, V),
            y.reshape(-1)
        )

        opt.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        opt.step()

        if step % 5 == 0:
            print(f"step {step} | loss {loss.item():.6f}")

    print("Training finished (DIINA-15 debug run).")


if __name__ == "__main__":
    run_debug()
