import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. معماری Adaptive-LIG v2.1 (تزریق ویژگی‌های آماری پیشین)
# =====================================================================
class AdaptiveLIG_v2_1(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        # شاخه تحلیل برجستگی: hidden_dim به همراه ۱ بعد واریانس لحظه‌ای
        self.saliency_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        # آستانه نویز آماری قابل تنظیم/یادگیری
        self.noise_threshold = nn.Parameter(torch.tensor(2.0))

    def forward(self, h: torch.Tensor, raw_signal: torch.Tensor) -> torch.Tensor:
        # محاسبه واریانس لحظه‌ای در راستای ابعاد فیچر ورودی خام
        batch_var = torch.var(raw_signal, dim=-1, keepdim=True)

        # تزریق واریانس خام به فضای نهفته
        combined = torch.cat([h, batch_var], dim=-1)
        s_t = self.saliency_net(combined)

        # ماسک بازداری آماری: اگر واریانس فراتر از آستانه رود، ماسک به صفر میل می‌کند
        noise_mask = torch.sigmoid((self.noise_threshold - batch_var) * 2.0)

        g_t = s_t * noise_mask
        return g_t


# =====================================================================
# 2. معماری DIINA-Robust-v2_1
# =====================================================================
class DIINA_Robust_v2_1(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.acoustic_in = nn.Linear(input_dim, hidden_dim)
        self.fast_path = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.slow_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        self.lig_v2_1 = AdaptiveLIG_v2_1(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor):
        x_proj = self.acoustic_in(x)
        h_fast, _ = self.fast_path(x_proj)

        # ارسال هم فیچرهای استخراجی و هم سیگنال خام برای تفکیک آماری
        g_t = self.lig_v2_1(h_fast, x)

        attn_out, _ = self.slow_attention(x_proj, x_proj, x_proj)
        h_slow = x_proj + attn_out

        # تلفیق مهارشده: نویز بالا => g_t میل به 0 => تکیه کامل بر Fast Path
        h_fused = (1.0 - g_t) * h_fast + g_t * h_slow
        pooled = torch.mean(h_fused, dim=1)
        return self.classifier(pooled), g_t


# =====================================================================
# 3. تست سنجش تفکیک آماری (Sanity Benchmark)
# =====================================================================
if __name__ == "__main__":
    print("==================================================================")
    print("   DIINA-Shield v2.1: Statistical Prior Integration Test")
    print("==================================================================")

    torch.manual_seed(42)
    model = DIINA_Robust_v2_1(input_dim=64, hidden_dim=64)

    # ۱. سیگنال زبانی ساختاریافته (واریانس کنترل‌شده حول مقدار میانگین)
    linguistic_signal = torch.randn(1, 100, 64) * 0.5 + 1.0

    # ۲. نویز تصادفی شدید با واریانس بسیار بالا
    extreme_noise = torch.randn(1, 100, 64) * 10.0

    model.eval()
    with torch.no_grad():
        _, gate_ling = model(linguistic_signal)
        avg_gate_ling = gate_ling.mean().item()

        _, gate_noise = model(extreme_noise)
        avg_gate_noise = gate_noise.mean().item()

    print("\n[+] Experimental Gating Output:")
    print(f"    - Linguistic Signal -> LIG v2.1 Activation (g_t): {avg_gate_ling:.4f}")
    print(f"    - Extreme Noise     -> LIG v2.1 Activation (g_t): {avg_gate_noise:.4f}")
    print("------------------------------------------------------------------")

    separation_ratio = avg_gate_ling / max(avg_gate_noise, 1e-6)
    print(f"    - Dynamic Separation Ratio: {separation_ratio:.2f}x")

    if separation_ratio >= 3.0:
        print("\n[SUCCESS]: Statistical Prior achieved decisive noise separation!")
    else:
        print("\n[TWEAK NEEDED]: Separation ratio below target.")
    print("==================================================================")
