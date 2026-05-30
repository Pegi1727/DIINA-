# DIINA-
```markdown
# DIINA: Inhibitory Neural Architecture for Afrikaans
‌
Official implementation of the DIINA model, an inhibitory neural architecture designed for double negation resolution and lexical ambiguity reduction in context-rich languages like Afrikaans.
‌
## Abstract
Afrikaans presents a compelling test case for computational models of negation: it is morphologically streamlined yet relies heavily on context and syntactic licensing. DIINA operationalizes inhibitory control as a learned mechanism that suppresses competing interpretations when negation is structurally licensed.
‌
## Key Performance Metrics
- Double Negation Accuracy: 94%
- Lexical Ambiguity Error Rate: 6%
- Inhibition Score (Secondary Negation): 0.92
‌
## Core Architecture
The model is built on a gated inhibitory framework where:

Contextual Encoding: Features are extracted via dense blocks.
Inhibitory Gating: A learned suppression function (InhibitoryGate) reduces activation of incompatible hypotheses.
Progressive Refinement: Inhibition intensifies in later layers to consolidate semantic interpretation.
‌
## Repository Structure
- `model.py`: The core PyTorch implementation of the DIINA architecture and Inhibitory Gate.
- `visualization.py`: Tools for generating inhibitory heatmaps and interpretability plots.
‌
## Citation
If you use this model or the findings in your research, please cite our article:
"Linguistic Efficiency and Inhibitory Control: Evaluating the DIINA Model on Afrikaans as a Morphologically Simplified but Context-Rich Language"
```
