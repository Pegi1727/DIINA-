import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. معماری Adaptive-LIG (نسخه ۲: سپر هوشمند) ---
class AdaptiveLIG(nn.Module):
    """
    نسخه تکامل‌یافته LIG که قادر است بین 'تضاد زبانی' و 'نویز تصادفی' تفکیک قائل شود.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        # شاخه تشخیص برجستگی زبانی (Linguistic Saliency)
        self.saliency_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        # شاخه تشخیص نویز (Noise Detector / Variance Estimation)
        self.noise_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        s_t = self.saliency_net(h)       # احتمال اهمیت واج‌شناختی
        n_t = self.noise_detector(h)     # برآورد نویز غیرساختاریافته
        
        # منطق بازدارندگی: نویز بالا، دسترسی به مسیر کند را بلوکه می‌کند
        g_t = s_t * (1.0 - n_t)
        return g_t


# --- 2. معماری DIINA-v2 (The Robust Edition) ---
class DIINA_Robust(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.acoustic_in = nn.Linear(input_dim, hidden_dim)

        # Fast Path (GRU پایدار و موضعی)
        self.fast_path = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        # Slow Path (Multi-Head Attention برای بافتار سراسری)
        self.slow_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        # گیت تطبیقی نسخه ۲
        self.lig_v2 = AdaptiveLIG(hidden_dim)

        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor):
        x_proj = self.acoustic_in(x)
        h_fast, _ = self.fast_path(x_proj)

        # فعال‌سازی گیت مهارکننده
        g_t = self.lig_v2(h_fast)

        attn_out, _ = self.slow_attention(x_proj, x_proj, x_proj)
        h_slow = x_proj + attn_out

        # ترکیب دینامیک: در شرایط نویز، g_t کوچک شده و اتکا به Fast Path بیشتر می‌شود
        h_fused = (1.0 - g_t) * h_fast + g_t * h_slow

        pooled = torch.mean(h_fused, dim=1)
        return self.classifier(pooled), g_t


# --- 3. بنچمارک تشخیصی (Sanity Check) ---
if __name__ == "__main__":
    print("==================================================================")
    print("   DIINA-Shield v2: Adaptive-LIG Discrimination Test")
    print("==================================================================")
    print("[+] Initializing Adaptive-LIG v2 architecture...")

    torch.manual_seed(42)
    model_v2 = DIINA_Robust(input_dim=64, hidden_dim=64)

    # ۱. سیگنال زبانی با تغییر تون معنادار (ساختاریافته)
    clean_linguistic = torch.randn(1, 100, 64) + 2.0 

    # ۲. نویز تصادفی شدید با واریانس بالا (غیرساختاریافته)
    noisy_signal = torch.randn(1, 100, 64) * 5.0

    model_v2.eval()
    with torch.no_grad():
        _, gate_ling = model_v2(clean_linguistic)
        avg_gate_ling = gate_ling.mean().item()

        _, gate_noise = model_v2(noisy_signal)
        avg_gate_noise = gate_noise.mean().item()

    print("\n[+] Verification Results:")
    print(f"    - Linguistic Pattern  -> Gate Activation (g_t): {avg_gate_ling:.4f}")
    print(f"    - High Gaussian Noise -> Gate Activation (g_t): {avg_gate_noise:.4f}")
    print("------------------------------------------------------------------")

    if avg_gate_ling > avg_gate_noise * 1.5:
        print("[STATUS: PASS] Adaptive-LIG successfully suppressed random perturbation!")
    else:
        print("[STATUS: CALIBRATION NEEDED] Network weights need calibration for wider separation.")
    print("==================================================================")
