"""
DIINA-Explorer: Unsupervised Acoustic Saliency and Boundary Discovery
For Unwritten / Endangered Tonal Languages
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. معماری DIINA Explorer ---
class DIINA_Explorer(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)

        # Fast Path (Global smooth contours)
        self.fast_path = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Slow Path (High-resolution multi-head attention)
        self.slow_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        # Latent Inhibitory Gating (LIG Saliency Detector)
        self.lig_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Self-Supervised Reconstruction Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim)
        )

    def forward(self, x):
        h_proj = F.relu(self.proj(x))
        h_fast, _ = self.fast_path(h_proj)

        # محاسبه گیت LIG به عنوان شاخص برجستگی (Saliency)
        saliency = self.lig_gate(h_fast)  # (Batch, Seq_Len, 1)

        # هدایت به مسیر کند (Slow Attention)
        attn_out, _ = self.slow_attention(h_proj, h_proj, h_proj)
        h_slow = h_proj + attn_out

        # ترکیب دو مسیره با وزن مهارکننده LIG
        h_fused = (1.0 - saliency) * h_fast + saliency * h_slow

        # بازسازی سیگنال آکوستیک
        recon = self.decoder(h_fused)
        return recon, saliency

# --- 2. شبیه‌سازی سیگنال‌های آکوستیک پیوسته زبان ناشناخته ---
def generate_unsupervised_acoustic_corpus(num_samples=120, seq_len=150, feat_dim=32):
    X = []
    ground_truth_boundaries = []
    
    for _ in range(num_samples):
        # زمینه هارمونیک ملایم
        t = np.linspace(0, 4 * np.pi, seq_len)
        base = np.sin(t)[:, None] * np.ones((seq_len, feat_dim))
        noise = np.random.normal(0, 0.15, (seq_len, feat_dim))
        seq = base + noise
        
        # تزریق مرزهای واجی / جهش‌های تونال ناگهانی در فواصل تصادفی
        bounds = np.random.choice(range(20, seq_len - 20), size=3, replace=False)
        bounds.sort()
        mask = np.zeros(seq_len)
        
        for b in bounds:
            # افت یا خیز ناگهانی و تغییر طیف فرکانسی
            seq[b-2:b+3, :] += np.random.uniform(2.5, 4.0, size=(5, feat_dim))
            mask[b-2:b+3] = 1.0
            
        X.append(seq)
        ground_truth_boundaries.append(mask)

    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        torch.tensor(np.array(ground_truth_boundaries), dtype=torch.float32)
    )

# --- 3. اجرای آزمایش و آموزش خودنظارتی ---
if __name__ == "__main__":
    print("==================================================================")
    print("  DIINA-Explorer: Unsupervised Acoustic Boundary Discovery Bench  ")
    print("==================================================================")

    torch.manual_seed(42)
    np.random.seed(42)

    feat_dim = 32
    seq_len = 150
    X_train, Y_bounds = generate_unsupervised_acoustic_corpus(num_samples=160, seq_len=seq_len, feat_dim=feat_dim)

    model = DIINA_Explorer(input_dim=feat_dim, hidden_dim=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print("[+] Training DIINA-Explorer via Self-Supervised Reconstruction...")
    start_time = time.time()
    
    model.train()
    epochs = 8
    for ep in range(epochs):
        optimizer.zero_grad()
        recon, saliency = model(X_train)
        
        recon_loss = criterion(recon, X_train)
        # تشویق گیت به توازن و جلوگیری از اشباع کامل
        sparsity_loss = 0.05 * torch.mean(saliency)
        total_loss = recon_loss + sparsity_loss

        total_loss.backward()
        optimizer.step()
        print(f"    Epoch {ep+1}/{epochs} | Recon MSE: {recon_loss.item():.4f} | Mean Saliency: {saliency.mean().item():.3f}")

    train_latency = time.time() - start_time
    print(f"[+] Training completed in {train_latency:.2f} seconds.")

    # ارزیابی اکتشاف مرزها (Boundary Discovery Evaluation)
    model.eval()
    with torch.no_grad():
        _, saliency = model(X_train)
        pred_saliency = saliency.squeeze(-1).numpy()
        gt_bounds = Y_bounds.numpy()

    # محاسبه همبستگی و دقت تشخیص مرزها بر اساس آستانه LIG
    threshold = np.percentile(pred_saliency, 85)
    pred_binary = (pred_saliency >= threshold).astype(float)
    
    tp = np.sum((pred_binary == 1.0) & (gt_bounds == 1.0))
    fp = np.sum((pred_binary == 1.0) & (gt_bounds == 0.0))
    fn = np.sum((pred_binary == 0.0) & (gt_bounds == 1.0))

    precision = tp / (tp + fp + 1e-8) * 100.0
    recall = tp / (tp + fn + 1e-8) * 100.0
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    print("\n================== DISCOVERY METRICS ==================")
    print(f"Boundary Precision : {precision:.2f}%")
    print(f"Boundary Recall    : {recall:.2f}%")
    print(f"F1 Boundary Score  : {f1:.2f}%")
    print(f"Pipeline Latency   : {train_latency:.2f}s")
    print("=======================================================")

    # ذخیره نتایج عددی در CSV
    df_results = pd.DataFrame([{
        "Task": "Unsupervised_Acoustic_Discovery",
        "Boundary_Precision_Pct": round(precision, 2),
        "Boundary_Recall_Pct": round(recall, 2),
        "Boundary_F1_Pct": round(f1, 2),
        "Recon_MSE": round(recon_loss.item(), 4),
        "Training_Latency_Sec": round(train_latency, 2)
    }])
    df_results.to_csv("yoruba_unsupervised_saliency.csv", index=False)
    print("[+] Results exported to 'yoruba_unsupervised_saliency.csv'.")

    # مصورسازی خروجی اکتشاف (Figure 13)
    sample_idx = 0
    s_curve = pred_saliency[sample_idx]
    gt_curve = gt_bounds[sample_idx]

    plt.figure(figsize=(10, 4.5), dpi=300)
    plt.plot(s_curve, color="#6a0dad", linewidth=2.0, label="DIINA LIG Saliency Profile $g(t)$")
    plt.fill_between(range(seq_len), 0, gt_curve, color="orange", alpha=0.35, label="True Phonetic Contrast / Boundary")
    plt.axhline(y=threshold, color="crimson", linestyle="--", alpha=0.8, label=f"Detection Threshold ({threshold:.2f})")
    plt.title("DIINA-Explorer: Zero-Shot Acoustic Saliency & Contrast Mapping", fontsize=12, pad=10)
    plt.xlabel("Time Frame (Acoustic Window)", fontsize=10)
    plt.ylabel("Inhibitory Saliency Activation", fontsize=10)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig("figure_13_unsupervised_saliency.png")
    plt.close()
    print("[+] Publication figure saved to 'figure_13_unsupervised_saliency.png'.")
