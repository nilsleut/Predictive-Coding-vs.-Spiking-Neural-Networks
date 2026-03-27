# Predictive Coding vs. Spiking Neural Networks — RSA on THINGS-fMRI

Representational Similarity Analysis (RSA) comparing a biologically motivated Predictive Coding network and a Spiking Neural Network against human fMRI responses from the THINGS dataset. Both models are built on top of ResNet-50 features and evaluated against cortical ROIs V1–IT.

---

## Key Results (N=3 subjects)

| Model | V1 | V4 | LOC | IT |
|---|---|---|---|---|
| PC r3 | 0.099 | 0.133 | 0.142 | **0.162** |
| SNN rate_l3 | 0.076 | 0.110 | 0.127 | 0.147 |
| ResNet-50 | 0.080 | 0.122 | 0.134 | 0.160 |

**PC r3 beats ResNet at LOC (+5.6%) and IT (+1.4%) across all three subjects.**

The central finding is the cross-over pattern in the PC network: r0 correlates strongly with early areas (V1: 0.30), while r3 reverses and correlates most strongly with late areas (IT: 0.16). This hierarchy is replicated in all three subjects and confirmed by a permutation test on the Layer×ROI interaction effect (observed Δ = +0.27, p = 0.007).

The SNN reaches 91% of PC r3 performance at IT but lacks the cross-over — all three layers show the same V1→IT gradient because they share the same layer4 input. The SNN result serves as a biologically plausible compression baseline, not a hierarchical model.

---

## Repository Structure

```
.
├── predictve_coding_v8.py   # PC network — training, RSA, group analysis, permutation test
├── snn_rsa_v3.py            # SNN — rate coding, RDM-MSE loss, group analysis
├── RSA_COMPARE_v2.ipynb     # Baseline comparison: ResNet-50, ViT-B/16, CLIP
└── README.md
```

---

## Methods

### Dataset
[THINGS-fMRI](https://things-initiative.org/) — 720 object concepts, 3 subjects (sub-01 to sub-03), ROIs: V1, V2, V3, V4, LOC, IT. Features extracted from ResNet-50 (ImageNet-pretrained), spatial dimensions collapsed via Global Average Pooling.

### Predictive Coding Network
3-layer PC network following Rao & Ballard (1999), trained unsupervised on ResNet-50 layer1–4 features. Each PC layer receives the corresponding ResNet layer as bottom-up constraint. Representations r0–r3 and error signals ε0–ε2 are evaluated via RSA. Training minimises Free Energy over T_infer=30 inference steps per stimulus, 100 epochs, early stopping (patience=15).

### Spiking Neural Network
3-layer network of Leaky Integrate-and-Fire neurons (snnTorch, β=0.9, threshold=0.1), trained via BPTT with surrogate gradients (fast sigmoid). Input: ResNet-50 layer4 features, min-max encoded to Bernoulli spike trains over T=50 timesteps. Training objective: RDM-MSE — batch-wise MSE between the SNN's output RDM and the ResNet-layer4 RDM. This directly optimises the RSA-relevant structure rather than absolute firing rates.

### RSA
Representational Dissimilarity Matrices (RDMs) computed via pairwise correlation distance. Model RDMs compared to fMRI RDMs using Spearman ρ on the upper triangle. Noise ceilings estimated via split-half Spearman-Brown (100 iterations). Bootstrap confidence intervals: 1000 resamples over stimulus pairs. Group-level significance: permutation test on the Layer×ROI interaction effect (N=1000 permutations, ROI labels shuffled).

---

## Installation

```bash
pip install torch torchvision snntorch numpy scipy matplotlib tqdm h5py pandas
```

Python 3.10+. GPU optional — both scripts run on CPU for N=720 stimuli.

---

## Usage

**PC network (all 3 subjects + group analysis):**
```bash
python predictve_coding_v8.py
```

**SNN (all 3 subjects + group analysis):**
```bash
python snn_rsa_v3.py
# or single subject:
python snn_rsa_v3.py --subjects sub-01
```

**With PC results for direct comparison:**
```bash
python snn_rsa_v3.py --pc-results outputs/pc_rho_results_sub-01.npy
```

Outputs are saved to:
- PC: `Predictive Coding/outputs/`
- SNN: `Spiking neural Networks/outputs_snn/`

---

## Outputs

| File | Description |
|---|---|
| `pc_rho_results_{sub}.npy` | Spearman ρ per layer (r0–r3, ε0–ε2) and ROI |
| `pc_noise_ceilings_{sub}.npy` | Split-half noise ceilings per ROI |
| `pc_hierarchy_{sub}.png` | Hierarchie gradient with 95% bootstrap CI |
| `pc_rsa_comparison_{sub}.png` | Full model comparison bar chart |
| `pc_group_rho.npy` | All subjects' rho_results |
| `pc_permutation_results.npy` | Observed interaction effect + p-value |
| `snn_rho_results_{sub}.npy` | Spearman ρ for rate/vmem/count × l1/l2/l3 |
| `snn_rsa_comparison_{sub}.png` | SNN hierarchy gradient + representation comparison |
| `snn_group_rho.npy` | All subjects' SNN rho_results |

---

## Interpretation

The PC result confirms the Rao & Ballard (1999) prediction: a hierarchical generative model trained on visual features, with top-down predictions and local Hebbian weight updates, develops layer-specific representations that align with the cortical hierarchy. Crucially, r0 maps to early visual areas (V1) and r3 maps to high-level areas (IT), with a systematic cross-over across layers.

The SNN result shows that biologically plausible spike-based coding alone is not sufficient to recover cortical hierarchy — the RDM-MSE loss successfully trains the SNN to approximate the ResNet-layer4 RDM structure (91% of PC r3 at IT), but without hierarchical inputs, all layers converge to the same IT-semantic representation. This suggests that temporal spike dynamics by themselves do not add structure beyond what is available in the feedforward features.

---

## References

Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience*, 2(1), 79–87.

Millidge, B., Seth, A., & Buckley, C. L. (2021). Predictive Coding: a Theoretical and Experimental Review. *arXiv:2107.12979*.

Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate gradient learning in spiking neural networks. *IEEE Signal Processing Magazine*, 36(6), 51–63.

Hebart, M. N., et al. (2023). THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior. *eLife*, 12, e82580.

---

## Author

Nils Leutenegger — upper secondary / pre-university project, Switzerland, 2026.
Built on the THINGS-fMRI dataset. ResNet-50 baseline from `RSA_COMPARE_v2.ipynb`.
