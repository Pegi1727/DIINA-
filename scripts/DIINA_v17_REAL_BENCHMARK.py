import os
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Reproducibility Setup
torch.manual_seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Running on device: {device}")

# -------------------------------------------------------------
# 1. Dataset Generation: SCAN-Lite
# -------------------------------------------------------------
COMMANDS = ["walk", "run", "jump", "look", "turn"]
ACTIONS = ["WALK", "RUN", "JUMP", "LOOK", "TURN"]
MODIFIERS = ["twice", "thrice"]

VOCAB = ["<pad>", "<sos>", "<eos>"] + COMMANDS + MODIFIERS + ACTIONS
w2i = {w: i for i, w in enumerate(VOCAB)}
i2w = {i: w for i, w in enumerate(VOCAB)}

pad_idx = w2i["<pad>"]
sos_idx = w2i["<sos>"]
eos_idx = w2i["<eos>"]


def generate_scan_sample():
    cmd = random.choice(COMMANDS)
    act = ACTIONS[COMMANDS.index(cmd)]
    mod = random.choice(["", "twice", "thrice"])

    inp = [cmd]
    if mod:
        inp.append(mod)

    out = []
    reps = 1
    if mod == "twice":
        reps = 2
    elif mod == "thrice":
        reps = 3
    out = [act] * reps

    return inp, out


raw_train = [generate_scan_sample() for _ in range(2000)]
raw_test = [generate_scan_sample() for _ in range(500)]


def encode(data, max_len=15):
    inputs, targets = [], []
    for inp, out in data:
        i_tokens = [sos_idx] + [w2i[w] for w in inp] + [eos_idx]
        o_tokens = [sos_idx] + [w2i[w] for w in out] + [eos_idx]

        i_tokens += [pad_idx] * (max_len - len(i_tokens))
        o_tokens += [pad_idx] * (max_len - len(o_tokens))

        inputs.append(i_tokens[:max_len])
        targets.append(o_tokens[:max_len])
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(
        targets, dtype=torch.long
    )


train_src, train_tgt = encode(raw_train)
test_src, test_tgt = encode(raw_test)

train_src, train_tgt = train_src.to(device), train_tgt.to(device)
test_src, test_tgt = test_src.to(device), test_tgt.to(device)

# -------------------------------------------------------------
# 2. Architectures
# -------------------------------------------------------------
# (A) Standard Transformer Baseline
class TransformerBaseline(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos = nn.Parameter(torch.randn(1, 100, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        b, seq_len = x.shape
        h = self.embed(x) + self.pos[:, :seq_len, :]
        mask = x == pad_idx
        out = self.encoder(h, src_key_padding_mask=mask)
        return self.fc_out(out)


# (B) DIINA Architecture
class FastPath(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x):
        out, _ = self.rnn(x)
        return out


class SlowPath(nn.Module):
    def __init__(self, d_model, nhead=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )

    def forward(self, x, mask=None):
        out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return out


class DIINA(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos = nn.Parameter(torch.randn(1, 100, d_model))

        self.fast_path = FastPath(d_model)
        self.slow_path = SlowPath(d_model, nhead)

        # Dynamic Inhibitory Gate
        self.gate = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        b, seq_len = x.shape
        h = self.embed(x) + self.pos[:, :seq_len, :]
        mask = x == pad_idx

        # Compute dynamic inhibition weights
        g = self.gate(h)  # Shape: (b, seq_len, 1)

        f_out = self.fast_path(h)
        s_out = self.slow_path(h, mask=mask)

        # Dual-stream routing
        out = (1.0 - g) * f_out + g * s_out
        return self.fc_out(out)


# -------------------------------------------------------------
# 3. Training & Evaluation Engine
# -------------------------------------------------------------
def train_model(model, src, tgt, epochs=15, lr=0.001, batch_size=64):
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n = len(src)

    history = {"loss": [], "time": []}
    start_time = time.time()

    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            b_src = src[idx]
            b_tgt = tgt[idx]

            optimizer.zero_grad()
            out = model(b_src)
            loss = criterion(out.view(-1, len(VOCAB)), b_tgt.view(-1))
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()

        avg_loss = ep_loss / (n // batch_size)
        history["loss"].append(avg_loss)
        history["time"].append(time.time() - start_time)
        print(f"Epoch {ep+1:02d}/{epochs:02d} | Loss: {avg_loss:.4f}")

    return history


def evaluate(model, src, tgt):
    model.eval()
    with torch.no_grad():
        out = model(src)
        preds = torch.argmax(out, dim=-1)

        mask = tgt != pad_idx
        correct = (preds == tgt) & mask
        acc = correct.sum().item() / mask.sum().item()
    return acc


# -------------------------------------------------------------
# 4. Run Benchmark & Comparison
# -------------------------------------------------------------
print("\n" + "=" * 50)
print("--- Training Transformer Baseline ---")
tf_model = TransformerBaseline(vocab_size=len(VOCAB)).to(device)
t0 = time.time()
tf_hist = train_model(tf_model, train_src, train_tgt, epochs=15)
tf_time = time.time() - t0
tf_acc = evaluate(tf_model, test_src, test_tgt)

print("\n" + "=" * 50)
print("--- Training DIINA Architecture ---")
diina_model = DIINA(vocab_size=len(VOCAB)).to(device)
t0 = time.time()
diina_hist = train_model(diina_model, train_src, train_tgt, epochs=15)
diina_time = time.time() - t0
diina_acc = evaluate(diina_model, test_src, test_tgt)

# -------------------------------------------------------------
# 5. Output Summary & Artifact Generation
# -------------------------------------------------------------
res_df = pd.DataFrame(
    {
        "Model": ["Transformer", "DIINA"],
        "Test Accuracy": [f"{tf_acc*100:.2f}%", f"{diina_acc*100:.2f}%"],
        "Training Time (s)": [f"{tf_time:.2f}", f"{diina_time:.2f}"],
        "Final Loss": [f"{tf_hist['loss'][-1]:.4f}", f"{diina_hist['loss'][-1]:.4f}"],
    }
)

print("\n" + "=" * 50)
print("BENCHMARK SUMMARY:")
print(res_df.to_string(index=False))
res_df.to_csv("DIINA_v17_REAL_RESULTS.csv", index=False)
print("\n[+] Saved results to DIINA_v17_REAL_RESULTS.csv")

# Plot Convergence Comparison
plt.figure(figsize=(8, 5))
plt.plot(tf_hist["loss"], label="Transformer", marker="o", color="#d9534f")
plt.plot(diina_hist["loss"], label="DIINA", marker="s", color="#0275d8")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Convergence Comparison: Transformer vs. DIINA")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("DIINA_v17_Result.png", dpi=300, bbox_inches="tight")
plt.close()
print("[+] Saved convergence plot to DIINA_v17_Result.png")
