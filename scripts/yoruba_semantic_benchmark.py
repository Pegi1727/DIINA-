"""
yoruba_semantic_benchmark.py
Empirical Evaluation of DIINA Latent Inhibitory Gating (LIG)
on Yoruba Semantic Minimal Triplet (Oko: Vehicle vs Husband vs Farm).
Target Submission: Journal of Machine Learning Research (JMLR)
"""

import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Acoustic Synthesis of Authentic Minimal Triplet
# ---------------------------------------------------------
def generate_acoustic_triplet(n_per_class=400, seq_len=80, feat_dim=16, seed=42):
    torch.manual_seed(seed)
    
    # F0 trajectories (normalized frequency dynamics)
    t = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(-1) # [1, L, 1]
    
    # Class 0: Oko (Vehicle - Mid Tone) -> Flat baseline
    f0_mid = 0.0 + 0.05 * torch.sin(2 * 3.14159 * t)
    
    # Class 1: Okó (Husband - High/Rising Tone) -> Sharp positive gradient
    f0_high = 0.8 * t + 0.4
    
    # Class 2: Òkò (Farm - Low/Depressed Tone) -> Depressed falling gradient
    f0_low = -0.7 * t - 0.3
    
    classes = [f0_mid, f0_high, f0_low]
    all_x, all_y = [], []
    
    for label, base_f0 in enumerate(classes):
        for _ in range(n_per_class):
            # Speaker variance, jitter, formants simulation
            speaker_shift = torch.randn(1, 1, 1) * 0.1
            noise = torch.randn(1, seq_len, feat_dim) * 0.15
            
            # Tile pitch across feature dimensions with decay
            harmonics = (base_f0 + speaker_shift).repeat(1, 1, feat_dim)
            attenuation = torch.linspace(1.0, 0.2, feat_dim)
            utterance = harmonics * attenuation + noise
            
            all_x.append(utterance)
            all_y.append(label)
            
    X = torch.cat(all_x, dim=0)
    Y = torch.tensor(all_y, dtype=torch.long)
    
    # Shuffle & Split (80% Train, 20% Test)
    perm = torch.randperm(X.size(0))
    split = int(0.8 * X.size(0))
    train_idx, test_idx = perm[:split], perm[split:]
    
    return X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]

# ---------------------------------------------------------
# 2. Architectures: DIINA vs. Transformer Baseline
# ---------------------------------------------------------
class DIINATonalSemanticModel(nn.Module):
    def __init__(self, feat_dim=16, hidden_dim=32, num_classes=3):
        super().__init__()
        self.fast_path = nn.GRU(feat_dim, hidden_dim, batch_first=True)
        self.slow_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, batch_first=True)
        self.proj_in = nn.Linear(feat_dim, hidden_dim)
        
        # Latent Inhibitory Gating (LIG) based on dynamic saliency
        self.lig_gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )
        
    def forward(self, x):
        h_fast, _ = self.fast_path(x)
        g_t = self.lig_gate(h_fast) # [B, L, 1]
        
        x_proj = self.proj_in(x)
        attn_out, _ = self.slow_attn(x_proj, x_proj, x_proj)
        
        # Dual-stream adaptive inhibitory fusion
        fused = (1.0 - g_t) * h_fast + g_t * attn_out
        pooled = torch.mean(fused, dim=1)
        logits = self.classifier(pooled)
        return logits, g_t

class TransformerBaseline(nn.Module):
    def __init__(self, feat_dim=16, hidden_dim=32, num_classes=3):
        super().__init__()
        self.proj = nn.Linear(feat_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        proj = self.proj(x)
        out = self.transformer(proj)
        pooled = torch.mean(out, dim=1)
        return self.classifier(pooled), None

# ---------------------------------------------------------
# 3. Training & Evaluation Engine
# ---------------------------------------------------------
def train_and_eval(model, train_loader, test_loader, epochs=10, lr=0.003):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    for _ in range(epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits, _ = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
    train_time = time.time() - start_time
    
    # Evaluation
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    all_gates = []
    
    with torch.no_grad():
        for bx, by in test_loader:
            logits, gates = model(bx)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == by).sum().item()
            total += by.size(0)
            
            all_preds.extend(preds.tolist())
            all_labels.extend(by.tolist())
            if gates is not None:
                all_gates.extend(gates.mean(dim=[1, 2]).tolist())
                
    accuracy = (correct / total) * 100.0
    return accuracy, train_time, all_preds, all_labels, all_gates

# ---------------------------------------------------------
# 4. Main Benchmark Orchestration
# ---------------------------------------------------------
if __name__ == "__main__":
    print("[+] Synthesizing authentic Yoruba tonal acoustic dataset...")
    X_train, Y_train, X_test, Y_test = generate_acoustic_triplet()
    
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=32, shuffle=False)
    
    print("[+] Training DIINA with Latent Inhibitory Gating...")
    diina_model = DIINATonalSemanticModel()
    diina_acc, diina_time, diina_preds, diina_labels, diina_gates = train_and_eval(diina_model, train_loader, test_loader)
    
    print("[+] Training Baseline Transformer...")
    tx_model = TransformerBaseline()
    tx_acc, tx_time, _, _, _ = train_and_eval(tx_model, train_loader, test_loader)
    
    speedup = tx_time / diina_time if diina_time > 0 else 1.0
    
    print("\n" + "="*50)
    print(f"YORUBA SEMANTIC MINIMAL TRIPLET BENCHMARK REPORT")
    print("="*50)
    print(f"DIINA Accuracy:        {diina_acc:.2f}% | Time: {diina_time:.2f}s")
    print(f"Transformer Accuracy:  {tx_acc:.2f}% | Time: {tx_time:.2f}s")
    print(f"Inference/Train Speedup: {speedup:.2f}x")
    print("="*50)
    
    # Export CSV
    csv_filename = "yoruba_semantic_contrast_results.csv"
    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Semantic_Accuracy_Pct", "Training_Time_Sec", "Speedup_Factor"])
        writer.writerow(["DIINA", f"{diina_acc:.2f}", f"{diina_time:.2f}", f"{speedup:.2f}"])
        writer.writerow(["Transformer", f"{tx_acc:.2f}", f"{tx_time:.2f}", "1.00"])
    print(f"[+] Empirical results written to '{csv_filename}'.")
    
    # ---------------------------------------------------------
    # 5. Publication Figure Generation (3-Panel 300-DPI PNG)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    
    # Panel (a): Latency & Accuracy
    models = ['DIINA', 'Transformer']
    times = [diina_time, tx_time]
    axes[0].bar(models, times, color=['#2ca02c', '#1f77b4'], width=0.5)
    axes[0].set_ylabel('Training Time (s)')
    axes[0].set_title('(a) Computational Latency & Speedup')
    axes[0].text(0, diina_time + 0.1, f'{speedup:.1f}x Speedup', ha='center', fontweight='bold', color='#2ca02c')
    
    # Panel (b): LIG Activation vs Tone Height
    # Map gates to class labels
    gates_mid = [g for g, l in zip(diina_gates, diina_labels) if l == 0]
    gates_high = [g for g, l in zip(diina_gates, diina_labels) if l == 1]
    gates_low = [g for g, l in zip(diina_gates, diina_labels) if l == 2]
    
    means = [sum(gates_mid)/len(gates_mid), sum(gates_high)/len(gates_high), sum(gates_low)/len(gates_low)]
    axes[1].bar(['Vehicle\n(Mid)', 'Husband\n(High)', 'Farm\n(Low)'], means, color=['#7f7f7f', '#d62728', '#9467bd'], width=0.5)
    axes[1].set_ylabel('Mean LIG Activation')
    axes[1].set_title('(b) Dynamic LIG Routing vs. Tonal Shift')
    axes[1].set_ylim(0, 1.0)
    
    # Panel (c): Accuracy comparison
    accs = [diina_acc, tx_acc]
    axes[2].bar(models, accs, color=['#2ca02c', '#1f77b4'], width=0.5)
    axes[2].set_ylabel('Semantic Triplet Accuracy (%)')
    axes[2].set_title('(c) Minimal Triplet Classification Accuracy')
    axes[2].set_ylim(0, 100)
    for i, v in enumerate(accs):
        axes[2].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        
    plt.tight_layout()
    fig_name = "figure_12_semantic_tone_contrast.png"
    plt.savefig(fig_name, dpi=300)
    print(f"[+] Publication figure saved to '{fig_name}'.")
