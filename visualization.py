python
import matplotlib.pyplot as plt
import seaborn as sns
import torch
‌
def plot_inhibitory_heatmap(gate_weights, layer_name="Inhibitory Gate"):
"""
Generates a heatmap of inhibitory weights to visualize
which features are being suppressed.
"""
weights = gate_weights.detach().cpu().numpy()
plt.figure(figsize=(10, 8))
sns.heatmap(weights[0, :20, 0, 0].reshape(4, 5), annot=True, cmap="YlGnBu")
plt.title(f"Inhibitory Activation Map - {layer_name}")
plt.xlabel("Feature Index")
plt.ylabel("Inhibition Intensity")
plt.show()
‌
# Example usage for the paper's figures
# weights = torch.rand(1, 64, 1, 1)
# plot_inhibitory_heatmap(weights)
