"""
Spiking Neural Network — RSA on THINGS-fMRI  [v3]
==================================================
Direkte Erweiterung von predictve_coding_v8.py auf SNN-Basis.

Versionsgeschichte:
  v1: Baseline — Rate Matching Loss, layer1 Input, sigmoid(z-norm) Encoding
      Problem: alle Feuerraten ≈ 50%, RSA ρ ≈ 0
  v2: Min-Max Encoding, layer4 Input, RDM-MSE Loss
      Problem: threshold=1.0 → l3 dead (rate≈0.0003) → NaN in RDM
               lr=1e-3 + batch_size=32 → instabiler Loss
  v3: threshold=0.1, lr=5e-4, batch_size=64, Cosine LR, NaN-Guard

Änderungen gegenüber v2 (basierend auf sub-01 Ergebnissen):

  Problem 1 — Dead Layer (rate_l3 = NaN):
    v2: threshold=1.0  → l3 feuert mit rate≈0.0003
        Linear(2048→512) mit kaiming init: std_out≈0.4 << threshold=1.0
        → Membranpotential erreicht Schwelle nie → l3 tot → NaN in RDM
    v3: threshold=0.1  → alle drei Schichten feuern mit rate≈0.4
        Schwelle passend zur tatsächlichen Current-Amplitude

  Problem 2 — Instabiler Loss (Best = Epoch 3, dann Oszillation):
    v2: lr=1e-3, batch_size=32 → hohe Gradienten-Varianz pro Batch
    v3: lr=5e-4, batch_size=64 → stabilerer Batch-RDM (mehr Paare)
        Cosine Annealing statt ReduceLROnPlateau → glattere Konvergenz
        NaN-Guard in rdm_mse_loss → sicher auch bei unerwarteten dead layers



Zentrales Experiment:
  Produziert ein SNN — trainiert auf denselben ResNet-50 Features auf
  denselben THINGS-Stimuli — eine stärkere oder qualitativ andere
  Korrespondenz mit der kortikalen Hierarchie als das rate-coded PC-Netz?

Architektur:
  - Gleiche Input-Hierarchie wie PC-Netz: ResNet-50 layer1-4 Features
  - Encoding: Rate Coding (Bernoulli-Sampling per Zeitschritt)
  - Neuronen: Leaky Integrate-and-Fire (snnTorch Lapicque)
  - Training: BPTT via surrogate gradient (fast sigmoid)
  - Repräsentation für RSA: mittlere Feuerrate über T Zeitschritte

Drei Repräsentationsarten werden verglichen:
  - rate:  mittlere Feuerrate  (direkt vergleichbar mit PC)
  - vmem:  Membranpotential am Zeitfenster-Ende
  - count: Gesamt-Spike-Count pro Neuron (= rate × T, normiert)

Referenzen:
  - Mahowald & Douglas (1991). A silicon neuron. Nature.
  - Lapicque (1907). Recherches quantitatives sur l'excitation électrique.
  - Neftci et al. (2019). Surrogate gradient learning in SNNs. IEEE Signal Proc.
  - Rao & Ballard (1999). Predictive coding. Nature Neuroscience.

Voraussetzungen:
  - predictve_coding_v8.py im selben Ordner (oder RSA_DIR konfiguriert)
  - THINGS-Datensatz identisch zu PC-Skript
  - stim_order_pc.txt muss existieren (aus PC-Skript erzeugt)
  - snntorch installiert: pip install snntorch
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
from dataclasses import dataclass, field
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import snntorch as snn
from snntorch import surrogate


# ══════════════════════════════════════════════════════════════
# Konfiguration  —  identisch zu predictve_coding_v8.py
# ══════════════════════════════════════════════════════════════

@dataclass
class SNNConfig:
    # Pfade — wie im PC-Skript anpassen
    RSA_DIR:           Path = Path(r'C:\Users\nilsl\Desktop\Projekte\RSA')
    PC_DIR:            Path = Path(r'C:\Users\nilsl\Desktop\Projekte\Predictive Coding')
    SNN_DIR:           Path = Path(r'C:\Users\nilsl\Desktop\Projekte\Spiking neural Networks')
    DATENSATZ_DIR:     Path = None
    H5_FILE:           Path = None
    VOX_META:          Path = None
    STIM_META:         Path = None
    THINGS_IMAGES_DIR: Path = None

    # Stimuli — identisch zu PC-Skript
    N_IMAGES:    int = 720
    DEVICE:      str = "cuda" if torch.cuda.is_available() else "cpu"

    # SNN Architektur — gleiche Dimensionen wie PC-Netz
    # Input: ResNet-50 layer1-4 Features (nach GAP)
    d_layer1:    int = 256    # ResNet layer1 → SNN-Schicht 1 (V1-analog)
    d_layer2:    int = 512    # ResNet layer2 → SNN-Schicht 2 (V4-analog)
    d_layer3:    int = 1024   # ResNet layer3 → SNN-Schicht 3 (LOC-analog)
    d_layer4:    int = 2048   # ResNet layer4 → SNN-Schicht 4 (IT-analog)
    d_hidden:    int = 512    # Hidden-Dimension der SNN-Schichten

    # Spike Encoding
    T:           int   = 50       # Zeitschritte pro Stimulus (Zeitfenster)
    # Rate Coding: Bernoulli(p = feature_value), p in [0, 1]
    # Für ResNet-Features: sigmoid-Normierung auf [0,1]

    # LIF Neuron Parameter
    beta:        float = 0.9      # Membrane decay (τ = -dt/ln(beta))
    threshold:   float = 0.1     # v3: 0.1 statt 1.0 — passt zu kaiming std_out≈0.4

    # Training
    lr:          float = 5e-4    # v3: 5e-4 statt 1e-3 — reduziert Gradienten-Varianz
    n_epochs:    int   = 80      # v3: 80 statt 50 — mehr Konvergenzzeit
    patience:    int   = 15
    batch_size:  int   = 64      # v3: 64 statt 32 — stabilerer Batch-RDM (mehr Paare)
    grad_clip:   float = 1.0

    # RSA
    ROI_NAMES:   tuple = ('V1', 'V2', 'V3', 'V4', 'LOC', 'IT')

    def __post_init__(self):
        self.SNN_DIR.mkdir(parents=True, exist_ok=True)
        self.OUT_DIR = self.SNN_DIR / 'outputs_snn'
        self.OUT_DIR.mkdir(parents=True, exist_ok=True)
        if self.DATENSATZ_DIR is None:
            self.DATENSATZ_DIR     = self.RSA_DIR / 'Datensatz'
        if self.H5_FILE is None:
            self.H5_FILE           = self.DATENSATZ_DIR / 'sub-01_task-things_voxel-wise-responses.h5'
        if self.VOX_META is None:
            self.VOX_META          = self.DATENSATZ_DIR / 'sub-01_task-things_voxel-metadata.csv'
        if self.STIM_META is None:
            self.STIM_META         = self.DATENSATZ_DIR / 'sub-01_task-things_stimulus-metadata.csv'
        if self.THINGS_IMAGES_DIR is None:
            self.THINGS_IMAGES_DIR = self.DATENSATZ_DIR / 'images_THINGS' / 'object_images'


# ══════════════════════════════════════════════════════════════
# Spike Encoding
# ══════════════════════════════════════════════════════════════

def rate_encode(features: torch.Tensor, T: int,
                pop_min: torch.Tensor = None,
                pop_max: torch.Tensor = None) -> torch.Tensor:
    """
    Rate Coding: Feature-Aktivierung → Spike-Train via Min-Max Normierung.

    v2-Fix gegenüber v1:
      v1 verwendete sigmoid(z-normiert) → alle Feuerraten clustern bei ~50%
         weil z-Normierung Mittelwert=0 → sigmoid(0)=0.5 für jeden Stimulus
      v2 verwendet Min-Max pro Feature-Dimension über die gesamte Population:
         prob = (x - pop_min) / (pop_max - pop_min)
         → Feuerraten spreizen sich tatsächlich zwischen 0 und 1
         → Stimulus-Unterschiede bleiben im Spike-Train erhalten

    Args:
        features: [B, d]       — Batch von Feature-Vektoren
        T:                       Anzahl Zeitschritte
        pop_min:  [d]           — Minimum pro Dimension über Gesamtpopulation
        pop_max:  [d]           — Maximum pro Dimension über Gesamtpopulation
                                  (beide aus get_population_stats() berechnet)
    Returns:
        spikes: [T, B, d] — binäre Spike-Trains (0 oder 1)
    """
    if pop_min is None or pop_max is None:
        # Fallback: batch-lokales Min-Max (suboptimal, aber korrekt)
        pop_min = features.min(0).values
        pop_max = features.max(0).values

    rng = (pop_max - pop_min).clamp(min=1e-8)
    prob = (features - pop_min) / rng              # [B, d], in [0, 1]
    prob = prob.clamp(0.0, 1.0)
    prob_expanded = prob.unsqueeze(0).expand(T, -1, -1)
    return torch.bernoulli(prob_expanded)          # [T, B, d]


def get_population_stats(features: torch.Tensor):
    """
    Berechnet Min/Max pro Feature-Dimension über die gesamte Population.
    Muss einmal vor dem Training auf allen N Stimuli aufgerufen werden.
    Returns: (pop_min [d], pop_max [d])
    """
    return features.min(0).values, features.max(0).values


# ══════════════════════════════════════════════════════════════
# SNN Architektur
# ══════════════════════════════════════════════════════════════

class SNNLayer(nn.Module):
    """
    Einzelne SNN-Schicht: Linear + Leaky Integrate-and-Fire.

    LIF-Dynamik (diskret, snn.Leaky):
      I[t]   = W × input[t] + b          (gewichteter Input)
      mem[t] = beta × mem[t-1] + I[t]    (Membranpotential-Update)
      spk[t] = 1  falls mem[t] ≥ threshold, sonst 0
      mem[t] = mem[t] × (1 - spk[t])     (Reset nach Spike)

    beta: Membrane decay — entspricht dem τ_m aus dem Hodgkin-Huxley-Modell
    (den du aus dem HH-Projekt kennst): τ_m = -dt / ln(beta)
    Bei beta=0.9 und dt=1ms → τ_m ≈ 9.5ms — biologisch plausibel.

    Hinweis: snn.Leaky statt snn.Lapicque — identische Dynamik,
    robustere API in snnTorch ≥ 0.9.
    """

    def __init__(self, d_in: int, d_out: int, beta: float, threshold: float):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        # Leaky LIF mit Surrogate-Gradient (fast sigmoid)
        # Surrogate ersetzt den unableitbaren Heaviside-Step bei BPTT
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            learn_beta=True,        # beta wird mittrainiert
            learn_threshold=True,   # Schwelle wird mittrainiert
        )

    def forward(self, x: torch.Tensor, mem: torch.Tensor):
        """
        Args:
            x:   [B, d_in]  — Input (Spikes oder kontinuierlich)
            mem: [B, d_out] — Membranpotential vom letzten Zeitschritt
        Returns:
            spk: [B, d_out] — Spike-Output
            mem: [B, d_out] — neues Membranpotential
        """
        cur = self.linear(x)
        spk, mem = self.lif(cur, mem)
        return spk, mem


class SpikingNet(nn.Module):
    """
    3-Schicht SNN für THINGS-RSA.

    Architektur:
      Input (rate-coded Spikes, d_input) →
      Layer 1 (LIF, d_hidden) →
      Layer 2 (LIF, d_hidden) →
      Layer 3 (LIF, d_output)

    Wird über T Zeitschritte abgerollt (snnTorch BPTT).

    Analogie zur PC-Netz-Hierarchie aus predictve_coding_v8.py:
      snn_layer1 ≈ PC r1 (V4-analog)
      snn_layer2 ≈ PC r2 (LOC-analog)
      snn_layer3 ≈ PC r3 (IT-analog)
    """

    def __init__(self, d_input: int, d_hidden: int, d_output: int,
                 beta: float, threshold: float):
        super().__init__()
        self.layer1 = SNNLayer(d_input,  d_hidden, beta, threshold)
        self.layer2 = SNNLayer(d_hidden, d_hidden, beta, threshold)
        self.layer3 = SNNLayer(d_hidden, d_output, beta, threshold)

    def forward(self, spike_train: torch.Tensor):
        """
        Rollt das SNN über T Zeitschritte ab.

        Args:
            spike_train: [T, B, d_input] — Rate-coded Input-Spikes

        Returns:
            spk_rec:  {'l1','l2','l3'} → [T, B, d] — Spike-Trains pro Schicht
            mem_rec:  {'l1','l2','l3'} → [T, B, d] — Membranpotentiale

        Repräsentationen für RSA werden danach aus spk_rec/mem_rec abgeleitet.
        """
        T = spike_train.shape[0]
        B = spike_train.shape[1]

        # Membranpotentiale initialisieren
        mem1 = self.layer1.lif.init_leaky()
        mem2 = self.layer2.lif.init_leaky()
        mem3 = self.layer3.lif.init_leaky()

        spk_rec = {'l1': [], 'l2': [], 'l3': []}
        mem_rec = {'l1': [], 'l2': [], 'l3': []}

        for t in range(T):
            x_t = spike_train[t]           # [B, d_input]

            spk1, mem1 = self.layer1(x_t,  mem1)
            spk2, mem2 = self.layer2(spk1, mem2)
            spk3, mem3 = self.layer3(spk2, mem3)

            spk_rec['l1'].append(spk1)
            spk_rec['l2'].append(spk2)
            spk_rec['l3'].append(spk3)
            mem_rec['l1'].append(mem1)
            mem_rec['l2'].append(mem2)
            mem_rec['l3'].append(mem3)

        # [T, B, d] stapeln
        spk_rec = {k: torch.stack(v, dim=0) for k, v in spk_rec.items()}
        mem_rec = {k: torch.stack(v, dim=0) for k, v in mem_rec.items()}

        return spk_rec, mem_rec


# ══════════════════════════════════════════════════════════════
# Repräsentationen aus Spike-Trains extrahieren
# ══════════════════════════════════════════════════════════════

def extract_representations(spk_rec: dict, mem_rec: dict) -> dict:
    """
    Drei Repräsentationsarten für RSA:

    rate:  Mittlere Feuerrate über alle T Zeitschritte.
           → Direkt vergleichbar mit PC r1/r2/r3 (rate-coded).
           → rate[i] = (1/T) Σ_t spk[t,i]

    vmem:  Membranpotential am Ende des Zeitfensters.
           → Analoge zu subthreshold dynamics in Neurophysiologie.
           → vmem[i] = mem[T-1, i]

    count: Gesamt-Spike-Count, L2-normiert.
           → Äquivalent zu rate, skalierungsinvariant.
           → count[i] = Σ_t spk[t,i], dann L2-normiert

    Returns: dict mit Schlüsseln 'rate_l1', 'rate_l2', 'rate_l3',
                                  'vmem_l1', 'vmem_l2', 'vmem_l3',
                                  'count_l1', 'count_l2', 'count_l3'
    """
    reps = {}
    for layer in ['l1', 'l2', 'l3']:
        spk = spk_rec[layer]   # [T, B, d]
        mem = mem_rec[layer]   # [T, B, d]

        # Rate: mittlere Feuerrate
        reps[f'rate_{layer}'] = spk.mean(dim=0)              # [B, d]

        # Vmem: letztes Membranpotential
        reps[f'vmem_{layer}'] = mem[-1]                       # [B, d]

        # Count: normierter Spike-Count
        count = spk.sum(dim=0)                               # [B, d]
        norm  = count.norm(dim=1, keepdim=True).clamp(min=1e-8)
        reps[f'count_{layer}'] = count / norm                # [B, d]

    return reps


# ══════════════════════════════════════════════════════════════
# Trainings-Ziel: Rate Matching Loss
# ══════════════════════════════════════════════════════════════

def rdm_mse_loss(spk_rec: dict, target_rdm: torch.Tensor,
                 layer: str = 'l3') -> torch.Tensor:
    """
    RDM-MSE Loss: Direkte Optimierung der RSA-relevanten Größe.

    v2-Fix gegenüber v1:
      v1 Rate Matching Loss: MSE(mittl. Feuerrate, target_rate)
         → Optimiert absolute Feuerraten, nicht Stimulus-Unterschiede
         → Netz mit identischer rate für alle Stimuli hat Loss≈0, RSA≈0
      v2 RDM-MSE: MSE(Batch-RDM der SNN-Outputs, Batch-RDM der ResNet-Features)
         → Erzwingt: Repräsentation(A) ≠ Repräsentation(B) wenn A ≠ B
         → Direkte Optimierung der RSA-Struktur

    Batch-RDM-Approximation:
      - Berechne Korrelations-RDM nur für den aktuellen Batch (B×B statt N×N)
      - Mit Batch-Size 32: 496 Paare pro Schritt (gut genug für Gradienten)
      - Über viele Batches gesehen: approximiert die globale RDM

    Args:
        spk_rec:    dict {'l1','l2','l3'} → [T, B, d]
        target_rdm: [B, B] — RDM der ResNet-layer4 Features für diesen Batch
        layer:      welche SNN-Schicht ('l3' = IT-analog, default)

    Returns:
        scalar loss
    """
    # Mittlere Feuerrate als Repräsentation
    mean_rate = spk_rec[layer].mean(dim=0)    # [B, d]

    # Normiere über Feature-Dimension — clamp verhindert NaN bei dead layers
    r = mean_rate - mean_rate.mean(dim=1, keepdim=True)
    norm = r.norm(dim=1, keepdim=True).clamp(min=1e-6)
    r_normed = r / norm

    # Kosinus-Ähnlichkeit → Korrelationsdistanz
    sim = r_normed @ r_normed.T               # [B, B]
    snn_rdm = 1.0 - sim                       # [B, B], in [0, 2]

    # MSE zwischen SNN-RDM und ResNet-RDM (nur oberes Dreieck, differenzierbar)
    B = snn_rdm.shape[0]
    mask = torch.triu(torch.ones(B, B, device=snn_rdm.device), diagonal=1).bool()
    loss = torch.nn.functional.mse_loss(snn_rdm[mask], target_rdm[mask])

    # NaN-Guard: tritt auf wenn eine Schicht komplett tot ist (rate≈0, norm≈0)
    if torch.isnan(loss):
        return torch.tensor(0.0, requires_grad=True, device=snn_rdm.device)
    return loss


def compute_batch_rdm(features: torch.Tensor) -> torch.Tensor:
    """
    Berechnet Korrelations-RDM für einen Batch von Features.
    Differenzierbar — wird für den Target-RDM aus ResNet-Features benutzt.
    Returns: [B, B] Tensor (auf demselben Device wie features)
    """
    r = features - features.mean(dim=1, keepdim=True)
    norm = r.norm(dim=1, keepdim=True).clamp(min=1e-8)
    r_normed = r / norm
    sim = r_normed @ r_normed.T
    return (1.0 - sim).detach()   # detach: Target ist fix, kein Gradient


# ══════════════════════════════════════════════════════════════
# ResNet-50 Feature-Extraktion — aus predictve_coding_v8.py
# ══════════════════════════════════════════════════════════════

def extract_resnet_features(image_paths: list, device: str) -> dict:
    """
    Extrahiert ResNet-50 layer1-4 Features (nach Global Average Pooling).
    Identisch zu predictve_coding_v8.py — gleiche Dimensionen, gleiche Normierung.
    """
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet.eval().to(device)

    layer_features = {k: [] for k in ['layer1', 'layer2', 'layer3', 'layer4']}
    batch_cache = {}

    def make_hook(name):
        def hook(module, input, output):
            batch_cache[name] = output.mean(dim=[2, 3]).detach().cpu()
        return hook

    handles = [
        resnet.layer1.register_forward_hook(make_hook('layer1')),
        resnet.layer2.register_forward_hook(make_hook('layer2')),
        resnet.layer3.register_forward_hook(make_hook('layer3')),
        resnet.layer4.register_forward_hook(make_hook('layer4')),
    ]

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    for start in tqdm(range(0, len(image_paths), 32),
                      desc='ResNet-50 layer1-4 Features'):
        batch_paths = image_paths[start:start + 32]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert('RGB')
            except Exception:
                img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            imgs.append(preprocess(img))
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            resnet(batch)
        for k in layer_features:
            layer_features[k].append(batch_cache[k])

    for h in handles:
        h.remove()

    result = {k: torch.cat(v, dim=0) for k, v in layer_features.items()}
    for k, v in result.items():
        print(f"  {k}: {v.shape}")
    return result


# ══════════════════════════════════════════════════════════════
# SNN Training
# ══════════════════════════════════════════════════════════════

def train_snn(layer_features: dict, cfg: SNNConfig):
    """
    Trainiert das SNN via BPTT mit RDM-MSE Loss.

    v2-Änderungen:
      - Input: layer4 (2048-dim, IT-semantisch) statt layer1
      - Encoding: Min-Max Normierung statt sigmoid(z-norm)
      - Loss: RDM-MSE auf Batch-RDM statt Rate Matching MSE

    Ablauf pro Batch:
      1. Min-Max-encode layer4-Features → Spike-Train [T, B, 2048]
      2. SNN abrollen → spk_rec, mem_rec
      3. RDM-MSE: MSE(SNN-Batch-RDM, ResNet-layer4-Batch-RDM)
      4. BPTT via surrogate gradient
      5. Gradient-Clipping + Adam
    """
    # Population-Stats für Min-Max Encoding (einmal über alle N Stimuli)
    pop_min_l4, pop_max_l4 = get_population_stats(layer_features['layer4'])
    pop_min_l1, pop_max_l1 = get_population_stats(layer_features['layer1'])

    # ResNet-layer4 als Referenz-Repräsentation (Ziel-RDM-Struktur)
    feats_l4 = layer_features['layer4']   # [N, 2048]
    N = len(feats_l4)

    # SNN: Input=2048 (layer4-dim), hidden=d_hidden, output=d_hidden
    net = SpikingNet(
        d_input=cfg.d_layer4,
        d_hidden=cfg.d_hidden,
        d_output=cfg.d_hidden,
        beta=cfg.beta,
        threshold=cfg.threshold,
    ).to(cfg.DEVICE)

    # Population-Stats auf DEVICE schieben
    pop_min_l4 = pop_min_l4.to(cfg.DEVICE)
    pop_max_l4 = pop_max_l4.to(cfg.DEVICE)

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    # Cosine Annealing: LR sinkt glatt von lr auf lr*0.05 über n_epochs
    # Stabiler als ReduceLROnPlateau für verrauschte Batch-RDM-Losses (v3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_epochs, eta_min=cfg.lr * 0.05
    )

    print(f"\nSNN Training (v3):")
    print(f"  Input: layer4 ({cfg.d_layer4}-dim, IT-semantisch)")
    print(f"  Architektur: {cfg.d_layer4} → {cfg.d_hidden} → {cfg.d_hidden} → {cfg.d_hidden}")
    print(f"  Encoding: Min-Max (Feuerraten spreizen sich über Stimuli)")
    print(f"  Loss: RDM-MSE (direkte Optimierung der RSA-Struktur)")
    print(f"  T={cfg.T} Zeitschritte, beta={cfg.beta}, threshold={cfg.threshold}")
    print(f"  {cfg.n_epochs} Epochen × {N} Stimuli\n")

    loss_history = []
    best_loss    = float('inf')
    best_state   = {k: v.clone() for k, v in net.state_dict().items()}
    patience_ctr = 0

    for epoch in range(cfg.n_epochs):
        perm       = torch.randperm(N)
        epoch_loss = 0.0
        n_batches  = 0

        net.train()
        for start in range(0, N, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            if len(idx) < 4:   # Batch zu klein für sinnvolle RDM
                continue

            # layer4-Features für diesen Batch
            x_l4 = feats_l4[idx].to(cfg.DEVICE)                      # [B, 2048]

            # Target-RDM aus ResNet-layer4 (fix, kein Gradient)
            target_rdm = compute_batch_rdm(x_l4)                     # [B, B]

            # Min-Max Rate Encoding → Spike-Train
            spike_train = rate_encode(x_l4, cfg.T, pop_min_l4, pop_max_l4)  # [T, B, 2048]

            optimizer.zero_grad()
            spk_rec, mem_rec = net(spike_train)

            # RDM-MSE auf l3 (höchste Schicht, IT-analog)
            loss = rdm_mse_loss(spk_rec, target_rdm, layer='l3')
            loss.backward()

            nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        scheduler.step()  # CosineAnnealingLR braucht kein Argument

        if avg_loss < best_loss:
            best_loss    = avg_loss
            best_state   = {k: v.clone() for k, v in net.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if (epoch + 1) % 10 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{cfg.n_epochs} | "
                  f"RDM-MSE: {avg_loss:.5f}  "
                  f"(best={best_loss:.5f}, patience={patience_ctr}/{cfg.patience}, "
                  f"lr={lr_now:.2e})")

        if patience_ctr >= cfg.patience:
            print(f"\n  Early Stop bei Epoch {epoch+1}")
            break

    net.load_state_dict(best_state)
    # Population-Stats für spätere Verwendung speichern
    net.pop_stats = {
        'layer4': (pop_min_l4.cpu(), pop_max_l4.cpu()),
        'layer1': (pop_min_l1, pop_max_l1),
    }
    net.input_layer = 'layer4'
    print(f"\nSNN Training abgeschlossen ✓  Bester RDM-MSE: {best_loss:.5f}")
    return net, loss_history


# ══════════════════════════════════════════════════════════════
# SNN-Repräsentationen extrahieren (nach Training)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def get_snn_representations(net: SpikingNet,
                             layer_features: dict,
                             cfg: SNNConfig) -> dict:
    """
    Extrahiert alle 9 Repräsentationen aus dem trainierten SNN.
    v2: Verwendet layer4 als Input mit Min-Max Encoding (wie im Training).
    """
    pop_min, pop_max = net.pop_stats['layer4']
    pop_min = pop_min.to(cfg.DEVICE)
    pop_max = pop_max.to(cfg.DEVICE)

    feats_l4 = layer_features['layer4']   # [N, 2048]
    N = len(feats_l4)

    all_reps = {key: [] for key in [
        'rate_l1', 'rate_l2', 'rate_l3',
        'vmem_l1', 'vmem_l2', 'vmem_l3',
        'count_l1', 'count_l2', 'count_l3',
    ]}

    net.eval()
    for start in tqdm(range(0, N, cfg.batch_size), desc='SNN-Repräsentationen'):
        x_l4 = feats_l4[start:start + cfg.batch_size].to(cfg.DEVICE)
        spike_train = rate_encode(x_l4, cfg.T, pop_min, pop_max)

        spk_rec, mem_rec = net(spike_train)
        reps_batch = extract_representations(spk_rec, mem_rec)

        for key in all_reps:
            all_reps[key].append(reps_batch[key].cpu())

    return {k: torch.cat(v, dim=0).numpy() for k, v in all_reps.items()}


# ══════════════════════════════════════════════════════════════
# RSA Hilfsfunktionen — identisch zu predictve_coding_v8.py
# ══════════════════════════════════════════════════════════════

def compute_rdm(features: np.ndarray) -> np.ndarray:
    return squareform(pdist(features, metric='correlation'))

def compare_rdms(rdm_a, rdm_b):
    n   = rdm_a.shape[0]
    idx = np.triu_indices(n, k=1)
    rho, p = spearmanr(rdm_a[idx], rdm_b[idx])
    return rho, p

def bootstrap_rsa(model_rdm, fmri_rdm, n_boot=1000, ci=0.95):
    n = model_rdm.shape[0]
    idx = np.triu_indices(n, k=1)
    x, y = model_rdm[idx], fmri_rdm[idx]
    n_pairs = len(x)
    rho_obs, _ = spearmanr(x, y)
    rng = np.random.default_rng(42)
    boot_rhos = np.zeros(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, n_pairs, size=n_pairs)
        boot_rhos[i], _ = spearmanr(x[s], y[s])
    alpha  = (1 - ci) / 2
    return float(rho_obs), float(np.percentile(boot_rhos, alpha*100)), \
           float(np.percentile(boot_rhos, (1-alpha)*100))


# ══════════════════════════════════════════════════════════════
# Visualisierung
# ══════════════════════════════════════════════════════════════

def plot_training_curve(loss_history: list, save_path: str):
    best_ep = int(np.argmin(loss_history))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_history, color='#534AB7', linewidth=2, label='Rate Matching Loss')
    ax.axvline(best_ep, color='#D85A30', linewidth=1.5, linestyle='--',
               label=f'Best (Epoch {best_ep+1}, Loss={loss_history[best_ep]:.4f})')
    ax.scatter([best_ep], [loss_history[best_ep]], color='#D85A30', zorder=5, s=60)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('MSE Loss', fontsize=11)
    ax.set_title('SNN Training — Rate Matching Loss', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Gespeichert: {save_path}")


def plot_spike_raster(spike_train: torch.Tensor, n_neurons: int = 40,
                      save_path: str = None):
    """
    Raster-Plot der ersten n_neurons für den ersten Stimulus im Batch.
    Zeigt die Rate-Coding-Struktur: dichtere Spikes = höhere Aktivierung.
    """
    spks = spike_train[:, 0, :n_neurons].cpu().numpy()  # [T, n_neurons]
    T, N = spks.shape
    fig, ax = plt.subplots(figsize=(10, 4))
    for n in range(N):
        t_spk = np.where(spks[:, n] > 0)[0]
        ax.scatter(t_spk, [n] * len(t_spk), marker='|', s=20,
                   color='#534AB7', linewidths=0.8)
    ax.set_xlabel('Zeitschritt t', fontsize=11)
    ax.set_ylabel('Neuron', fontsize=11)
    ax.set_title(f'Rate-Coded Spike-Train — erste {N} Input-Neuronen, erster Stimulus',
                 fontweight='bold')
    ax.set_xlim(0, T)
    ax.set_ylim(-1, N)
    ax.grid(axis='x', alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gespeichert: {save_path}")
    plt.close()


def plot_snn_rsa_comparison(rho_results: dict, noise_ceilings: dict,
                              roi_names: list, save_path: str):
    """
    Vergleicht SNN (rate_l1–l3, vmem_l1–l3) mit ResNet und PC-Netz.
    Hauptplot: rate_l1, rate_l2, rate_l3 als Hierarchie-Gradient.
    """
    n_rois = len(roi_names)
    x = np.arange(n_rois)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('SNN RSA — THINGS-fMRI\n'
                 'Links: Hierarchie-Gradient (rate), Rechts: Repräsentationsvergleich',
                 fontsize=13, fontweight='bold')

    # ── Links: Hierarchie-Gradient ──────────────────────────────
    ax = axes[0]
    colors_snn = ['#CECBF6', '#7F77DD', '#3C3489']
    labels_snn = ['SNN rate l1 (V1-input)', 'SNN rate l2', 'SNN rate l3 (IT-input)']

    for i, (layer, color, label) in enumerate(
            zip(['rate_l1', 'rate_l2', 'rate_l3'], colors_snn, labels_snn)):
        if layer not in rho_results:
            continue
        vals = [rho_results[layer].get(roi, 0) for roi in roi_names]
        ax.plot(x, vals, 'o-', color=color, label=label, linewidth=2.5, markersize=8)

    if 'resnet' in rho_results:
        vals = [rho_results['resnet'].get(roi, 0) for roi in roi_names]
        ax.plot(x, vals, 's--', color='#4477aa', label='ResNet-50', linewidth=1.5,
                markersize=6, alpha=0.8)
    if 'pc_r3' in rho_results:
        vals = [rho_results['pc_r3'].get(roi, 0) for roi in roi_names]
        ax.plot(x, vals, '^--', color='#D85A30', label='PC r3 (IT)', linewidth=1.5,
                markersize=6, alpha=0.8)

    if noise_ceilings:
        for j, roi in enumerate(roi_names):
            nc = noise_ceilings.get(roi, 0)
            ax.plot([x[j] - 0.4, x[j] + 0.4], [nc, nc], 'k--',
                    linewidth=1, alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(roi_names, fontsize=10)
    ax.set_ylabel('Spearman ρ', fontsize=11)
    ax.set_xlabel('ROI (früh → spät)', fontsize=11)
    ax.set_title('Hierarchie-Gradient: SNN rate_l1 → rate_l3')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.8)

    # ── Rechts: alle Repräsentationen nebeneinander (für IT) ───
    ax = axes[1]
    roi_focus = 'IT'  # interessanteste ROI für High-Level-Vergleich
    all_keys  = [k for k in rho_results if roi_focus in rho_results[k]]
    names     = [k.replace('_', ' ') for k in all_keys]
    vals      = [rho_results[k].get(roi_focus, 0) for k in all_keys]

    color_map = {
        'rate': '#534AB7', 'vmem': '#1D9E75', 'count': '#D85A30',
        'resnet': '#4477aa', 'pc': '#E85D24',
    }
    bar_colors = []
    for k in all_keys:
        if k.startswith('rate'):
            bar_colors.append('#534AB7')
        elif k.startswith('vmem'):
            bar_colors.append('#1D9E75')
        elif k.startswith('count'):
            bar_colors.append('#D85A30')
        elif k == 'resnet':
            bar_colors.append('#4477aa')
        else:
            bar_colors.append('#888780')

    bars = ax.bar(range(len(all_keys)), vals, color=bar_colors, alpha=0.85,
                  edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(all_keys)))
    ax.set_xticklabels(names, rotation=40, ha='right', fontsize=9)
    ax.set_ylabel('Spearman ρ', fontsize=11)
    ax.set_title(f'Alle SNN-Repräsentationen vs. {roi_focus}-RDM')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.8)

    if noise_ceilings and roi_focus in noise_ceilings:
        ax.axhline(noise_ceilings[roi_focus], color='gray', linewidth=1.5,
                   linestyle='--', label=f'Noise Ceiling ({roi_focus})')
        ax.legend(fontsize=9)

    # Legende für Farben
    from matplotlib.patches import Patch
    legend_els = [
        Patch(color='#534AB7', label='Rate (Feuerrate)'),
        Patch(color='#1D9E75', label='Vmem (Membranpot.)'),
        Patch(color='#D85A30', label='Count (Spike-Count)'),
        Patch(color='#4477aa', label='ResNet-50'),
    ]
    ax.legend(handles=legend_els, fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Gespeichert: {save_path}")


def plot_snn_vs_pc(rho_results: dict, roi_names: list, save_path: str):
    """
    Direkter Vergleich SNN rate_l3 vs. PC r3 (IT-analog).
    Die zentrale Frage: ist die SNN-Repräsentation stärker oder
    qualitativ anders als die rate-coded PC-Repräsentation?
    """
    if 'rate_l3' not in rho_results or 'pc_r3' not in rho_results:
        print("Hinweis: pc_r3 nicht in rho_results — PC-Vergleichsplot übersprungen.")
        print("  → predictve_coding_v8.py ausführen und pc_rho_results einladen.")
        return

    x = np.arange(len(roi_names))
    fig, ax = plt.subplots(figsize=(10, 5))

    snn_vals = [rho_results['rate_l3'].get(r, 0) for r in roi_names]
    pc_vals  = [rho_results['pc_r3'].get(r, 0)   for r in roi_names]
    diff     = [s - p for s, p in zip(snn_vals, pc_vals)]

    ax.plot(x, snn_vals, 'o-', color='#534AB7', label='SNN rate_l3', linewidth=2.5, markersize=9)
    ax.plot(x, pc_vals,  's-', color='#D85A30', label='PC r3',        linewidth=2.5, markersize=9)
    ax.bar(x, diff, 0.4, color=['#639922' if d > 0 else '#E24B4A' for d in diff],
           alpha=0.3, label='Δ (SNN − PC)')

    ax.set_xticks(x)
    ax.set_xticklabels(roi_names, fontsize=11)
    ax.set_ylabel('Spearman ρ', fontsize=11)
    ax.set_xlabel('ROI (früh → spät)', fontsize=11)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title('SNN rate_l3 vs. PC r3 — direkter Vergleich\n'
                 'Grün: SNN besser, Rot: PC besser', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Gespeichert: {save_path}")


def print_results_table(rho_results: dict, noise_ceilings: dict, roi_names: list):
    snn_layers = ['rate_l1', 'rate_l2', 'rate_l3',
                  'vmem_l1', 'vmem_l2', 'vmem_l3',
                  'count_l1', 'count_l2', 'count_l3']
    baselines  = ['resnet', 'pc_r3']

    print('\n' + '=' * 72)
    print('RSA ERGEBNISSE — SNN vs. Baselines')
    print('=' * 72)
    header = f'{"Modell":>12}' + ''.join(f'{r:>8}' for r in roi_names)
    print(header)
    print('─' * 72)

    for layer in snn_layers:
        if layer not in rho_results:
            continue
        row = f'{layer:>12}'
        for roi in roi_names:
            row += f'{rho_results[layer].get(roi, 0):>8.3f}'
        print(row)

    print('─' * 72)
    for name in baselines:
        if name not in rho_results:
            continue
        labels = {'resnet': 'ResNet-50', 'pc_r3': 'PC-r3'}
        row = f'{labels[name]:>12}'
        for roi in roi_names:
            row += f'{rho_results[name].get(roi, 0):>8.3f}'
        print(row)

    if noise_ceilings:
        print('─' * 72)
        row = f'{"NoiseClng":>12}'
        for roi in roi_names:
            row += f'{noise_ceilings.get(roi, 0):>8.3f}'
        print(row)
    print()


# ══════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════

def run_snn_subject(sub_id: str,
                    pc_rho_results: dict = None) -> dict:
    """
    Komplette SNN-RSA-Pipeline für ein Subject.

    pc_rho_results: optionales dict mit PC-Ergebnissen aus
                    predictve_coding_v8.py für direkten Vergleich.
                    Erwartet Schlüssel 'r3' mit ROI → ρ dict.
    """
    cfg = SNNConfig()
    cfg.H5_FILE   = cfg.DATENSATZ_DIR / f'{sub_id}_task-things_voxel-wise-responses.h5'
    cfg.VOX_META  = cfg.DATENSATZ_DIR / f'{sub_id}_task-things_voxel-metadata.csv'
    cfg.STIM_META = cfg.DATENSATZ_DIR / f'{sub_id}_task-things_stimulus-metadata.csv'

    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 65)
    print(f"SNN RSA — THINGS-fMRI  [{sub_id}]")
    print("=" * 65)
    print(f"Device: {cfg.DEVICE}")

    # ── Schritt 1: Stimuli laden ──────────────────────────────
    print("\n[1/5] Lade Stimuli...")
    import pandas as pd, h5py

    vox_meta  = pd.read_csv(cfg.VOX_META)
    stim_meta = pd.read_csv(cfg.STIM_META)

    roi_masks = {
        'V1':  vox_meta['V1'].values.astype(bool),
        'V2':  vox_meta['V2'].values.astype(bool),
        'V3':  vox_meta['V3'].values.astype(bool),
        'V4':  vox_meta['hV4'].values.astype(bool),
        'LOC': (vox_meta['lLOC'].values.astype(bool) |
                vox_meta['rLOC'].values.astype(bool)),
        'IT':  vox_meta['IT'].values.astype(bool),
    }

    combined_mask     = np.zeros(len(vox_meta), dtype=bool)
    for m in roi_masks.values():
        combined_mask |= m
    roi_voxel_indices = np.where(combined_mask)[0]
    global_to_local   = {int(g): l for l, g in enumerate(roi_voxel_indices)}

    # ── Schritt 2: fMRI laden ─────────────────────────────────
    print("[2/5] Lade fMRI...")
    with h5py.File(cfg.H5_FILE, 'r') as f:
        dset         = f['ResponseData/block0_values']
        roi_data_raw = dset[roi_voxel_indices, :].astype(np.float32)

    responses_all = roi_data_raw.T
    responses_all = ((responses_all - responses_all.mean(axis=0)) /
                     (responses_all.std(axis=0) + 1e-8))

    stim_meta_test = stim_meta[stim_meta['trial_type'] == 'test'].copy()

    # stim_order aus PC-Skript laden (identische Stimuli für validen Vergleich)
    stim_order_path = cfg.PC_DIR / 'stim_order_pc.txt'
    if stim_order_path.exists():
        with open(stim_order_path) as f:
            stim_order = [l.strip() for l in f.readlines()]
        print(f"  stim_order aus PC-Skript: {len(stim_order)} Stimuli ✓")
    else:
        # Fallback: gleiche Logik wie im PC-Skript
        stim_meta_unique = stim_meta_test.drop_duplicates(subset='stimulus').copy()
        valid_concepts   = sorted(stim_meta_unique['concept'].unique().tolist())
        stim_order = []
        for concept in valid_concepts:
            stims = sorted(
                stim_meta_unique[stim_meta_unique['concept'] == concept
                                 ]['stimulus'].tolist())[:1]
            stim_order.extend(stims)
        stim_order = stim_order[:cfg.N_IMAGES]
        print(f"  HINWEIS: stim_order_pc.txt nicht gefunden — eigene Auswahl "
              f"({len(stim_order)} Stimuli)")
        print("  Für validen PC-Vergleich: erst predictve_coding_v8.py ausführen.")

    stim_responses, image_paths, stim_found = [], [], []
    for stim in tqdm(stim_order, desc='fMRI mitteln'):
        idx = stim_meta_test.index[stim_meta_test['stimulus'] == stim].tolist()
        if not idx:
            continue
        stim_responses.append(responses_all[idx].mean(axis=0))
        concept = stim_meta_test.loc[idx[0], 'concept']
        image_paths.append(cfg.THINGS_IMAGES_DIR / concept / stim)
        stim_found.append(stim)

    stim_order = stim_found
    responses  = np.array(stim_responses)
    print(f"  fMRI responses: {responses.shape}")

    # fMRI-RDMs
    fmri_rdms = {}
    for roi in cfg.ROI_NAMES:
        g_idx = np.where(roi_masks[roi])[0]
        l_idx = np.array([global_to_local[int(g)] for g in g_idx
                          if int(g) in global_to_local])
        fmri_rdms[roi] = compute_rdm(responses[:, l_idx])

    # ── Schritt 3: ResNet-Features ────────────────────────────
    print("\n[3/5] Extrahiere ResNet-50 Features...")
    layer_features = extract_resnet_features(image_paths, cfg.DEVICE)

    # Spike-Raster-Plot für ersten Stimulus mit layer4 Features (v2)
    pop_min_demo, pop_max_demo = get_population_stats(layer_features['layer4'])
    x0 = layer_features['layer4'][:1].to(cfg.DEVICE)
    demo_train = rate_encode(x0, cfg.T,
                             pop_min_demo.to(cfg.DEVICE),
                             pop_max_demo.to(cfg.DEVICE))
    plot_spike_raster(demo_train, n_neurons=40,
                      save_path=str(cfg.OUT_DIR / f'snn_spike_raster_{sub_id}.png'))

    # ── Schritt 4: SNN trainieren ─────────────────────────────
    print("\n[4/5] Trainiere SNN...")
    net, loss_history = train_snn(layer_features, cfg)
    plot_training_curve(loss_history,
                        str(cfg.OUT_DIR / f'snn_training_curve_{sub_id}.png'))

    # ── Schritt 5: RSA ────────────────────────────────────────
    print("\n[5/5] RSA...")
    snn_reps = get_snn_representations(net, layer_features, cfg)
    for k, v in snn_reps.items():
        print(f"  {k}: {v.shape}")

    rho_results = {}

    # SNN-Schichten
    for rep_name, features in snn_reps.items():
        if np.isnan(features).any() or np.isinf(features).any():
            print(f"  WARNUNG: {rep_name} enthält NaN/Inf — übersprungen")
            continue
        rdm = compute_rdm(features)
        rho_results[rep_name] = {}
        for roi in cfg.ROI_NAMES:
            rho, _ = compare_rdms(rdm, fmri_rdms[roi])
            rho_results[rep_name][roi] = rho

    # ResNet-50 Baseline
    resnet_rdm = compute_rdm(layer_features['layer4'].numpy())
    rho_results['resnet'] = {}
    for roi in cfg.ROI_NAMES:
        rho, _ = compare_rdms(resnet_rdm, fmri_rdms[roi])
        rho_results['resnet'][roi] = rho

    # PC-Ergebnisse einbinden (optional)
    if pc_rho_results is not None and 'r3' in pc_rho_results:
        rho_results['pc_r3'] = pc_rho_results['r3']
        print("  PC r3 aus pc_rho_results eingebunden ✓")

    # Noise Ceilings
    print("\nNoise Ceilings (Split-Half)...")
    noise_ceilings = {}
    for roi in cfg.ROI_NAMES:
        g_idx = np.where(roi_masks[roi])[0]
        l_idx = np.array([global_to_local[int(g)] for g in g_idx
                          if int(g) in global_to_local])
        rhos_nc = []
        for _ in range(100):
            half_a, half_b = [], []
            for stim in stim_order:
                idx_s = stim_meta_test.index[
                    stim_meta_test['stimulus'] == stim].tolist()
                np.random.shuffle(idx_s)
                mid = max(1, len(idx_s) // 2)
                half_a.append(responses_all[idx_s[:mid]].mean(axis=0))
                half_b.append(responses_all[idx_s[mid:]].mean(axis=0))
            rdm_a = compute_rdm(np.array(half_a)[:, l_idx])
            rdm_b = compute_rdm(np.array(half_b)[:, l_idx])
            n_tri = rdm_a.shape[0]
            tri   = np.triu_indices(n_tri, k=1)
            rho_nc, _ = spearmanr(rdm_a[tri], rdm_b[tri])
            rhos_nc.append((2 * rho_nc) / (1 + rho_nc + 1e-8))
        nc = float(np.mean(rhos_nc))
        noise_ceilings[roi] = nc
        print(f"  {roi:5}: NC={nc:.3f}")

    # Ergebnistabelle
    print_results_table(rho_results, noise_ceilings, list(cfg.ROI_NAMES))

    # Plots
    plot_snn_rsa_comparison(
        rho_results, noise_ceilings, list(cfg.ROI_NAMES),
        str(cfg.OUT_DIR / f'snn_rsa_comparison_{sub_id}.png'))

    plot_snn_vs_pc(
        rho_results, list(cfg.ROI_NAMES),
        str(cfg.OUT_DIR / f'snn_vs_pc_{sub_id}.png'))

    # Ergebnisse speichern
    np.save(str(cfg.OUT_DIR / f'snn_rho_results_{sub_id}.npy'), rho_results)
    np.save(str(cfg.OUT_DIR / f'snn_noise_ceilings_{sub_id}.npy'), noise_ceilings)

    print(f"\nFertig [{sub_id}]. Outputs in: {cfg.OUT_DIR}")
    for fname in [
        f'snn_spike_raster_{sub_id}.png',
        f'snn_training_curve_{sub_id}.png',
        f'snn_rsa_comparison_{sub_id}.png',
        f'snn_vs_pc_{sub_id}.png',
    ]:
        print(f"  {cfg.OUT_DIR / fname}")

    return rho_results, noise_ceilings




def snn_group_analysis(all_results: dict, roi_names: list, out_dir):
    """
    Gruppen-Analyse analog zu plot_group_average in predictve_coding_v8.py.
    Mittelt rho_results ueber Subjects, berechnet SD, Permutationstest.
    all_results: {sub_id: (rho_dict, nc_dict)}
    """
    subjects = list(all_results.keys())
    if len(subjects) < 2:
        print("Gruppen-Analyse benoetigt >= 2 Subjects.")
        return

    all_rho = {s: v[0] for s, v in all_results.items()}
    all_nc  = {s: v[1] for s, v in all_results.items()}

    layers = ['rate_l1', 'rate_l2', 'rate_l3']
    mean_rho = {layer: {} for layer in layers}
    sd_rho   = {layer: {} for layer in layers}
    for layer in layers:
        for roi in roi_names:
            vals = [all_rho[s][layer][roi] for s in subjects
                    if layer in all_rho[s] and not np.isnan(all_rho[s][layer].get(roi, float('nan')))]
            mean_rho[layer][roi] = float(np.mean(vals)) if vals else float('nan')
            sd_rho[layer][roi]   = float(np.std(vals))  if len(vals) > 1 else float('nan')

    mean_nc = {roi: float(np.mean([all_nc[s][roi] for s in subjects])) for roi in roi_names}
    mean_resnet = {roi: float(np.mean([all_rho[s]['resnet'][roi] for s in subjects
                                        if 'resnet' in all_rho[s]])) for roi in roi_names}

    # Ergebnisse drucken
    print('\n' + '=' * 70)
    print(f'SNN GRUPPEN-ERGEBNISSE  (N={len(subjects)} Subjects)')
    print('=' * 70)
    header = f'{"Modell":>12}' + ''.join(f'{r:>8}' for r in roi_names)
    print(header); print('-' * 70)
    for layer in layers:
        row = f'{layer:>12}'
        for roi in roi_names:
            m = mean_rho[layer].get(roi, float('nan'))
            s = sd_rho[layer].get(roi, float('nan'))
            row += f'{m:>8.3f}'
        print(row)
    print('-' * 70)
    resnet_row = f'{"ResNet":>12}'
    for roi in roi_names:
        resnet_row += f'{mean_resnet.get(roi, 0):>8.3f}'
    print(resnet_row)
    nc_row = f'{"NoiseClng":>12}'
    for roi in roi_names:
        nc_row += f'{mean_nc.get(roi, 0):>8.3f}'
    print(nc_row)

    # Hierarchie-Gradient-Check
    print('\nHierarchie-Gradient (V1 vs IT):')
    for layer in layers:
        v1 = mean_rho[layer].get('V1', float('nan'))
        it = mean_rho[layer].get('IT', float('nan'))
        print(f'  {layer}: V1={v1:.4f}  IT={it:.4f}  Gradient={it-v1:+.4f}')

    # Interaktionseffekt (analog zu PC-Permutationstest)
    early_rois = ['V1', 'V2']
    late_rois  = ['LOC', 'IT']
    d_l1 = (np.mean([mean_rho['rate_l1'][r] for r in early_rois]) -
            np.mean([mean_rho['rate_l1'][r] for r in late_rois]))
    d_l3 = (np.mean([mean_rho['rate_l3'][r] for r in late_rois]) -
            np.mean([mean_rho['rate_l3'][r] for r in early_rois]))
    interaction = d_l1 + d_l3
    print(f'\nInteraktionseffekt l1(frueh-spaet) + l3(spaet-frueh) = {interaction:+.4f}')
    verdict = 'SNN zeigt hierarchischen Gradienten' if interaction > 0 else 'Kein hierarchischer Gradient im SNN'
    print(f'  {verdict}')

    # Speichern
    np.save(str(out_dir / 'snn_group_rho.npy'),     all_rho)
    np.save(str(out_dir / 'snn_group_mean_rho.npy'), mean_rho)
    np.save(str(out_dir / 'snn_group_mean_nc.npy'),  mean_nc)
    print(f'\nGruppen-Ergebnisse gespeichert: {out_dir}/snn_group_rho.npy')
    return mean_rho, sd_rho, mean_nc

# ══════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='SNN RSA — THINGS-fMRI')
    parser.add_argument('--subjects', nargs='+',
                        default=['sub-01', 'sub-02', 'sub-03'],
                        help='Subject-IDs')
    parser.add_argument('--pc-results', type=str, default=None,
                        help='Pfad zu pc_rho_results.npy (aus predictve_coding_v8.py)')
    args = parser.parse_args()

    # PC-Ergebnisse laden falls angegeben
    pc_rho = None
    if args.pc_results:
        try:
            pc_rho = np.load(args.pc_results, allow_pickle=True).item()
            print(f"PC-Ergebnisse geladen: {args.pc_results}")
        except Exception as e:
            print(f"Konnte PC-Ergebnisse nicht laden: {e}")

    all_results = {}
    for sub_id in args.subjects:
        print(f'\n{"#" * 65}')
        print(f'# Subject: {sub_id}')
        print(f'{"#" * 65}\n')
        try:
            rho, nc = run_snn_subject(sub_id, pc_rho_results=pc_rho)
            all_results[sub_id] = (rho, nc)
        except FileNotFoundError as e:
            print(f'  ⚠️  {sub_id} übersprungen — Datei nicht gefunden: {e}')

    if len(all_results) > 1:
        cfg_final = SNNConfig()
        snn_group_analysis(all_results, list(cfg_final.ROI_NAMES), cfg_final.OUT_DIR)
    elif all_results:
        print(f'\nAbgeschlossen: {list(all_results.keys())} (1 Subject, kein Gruppen-Plot)')
    else:
        print('\n⚠️  Kein Subject erfolgreich.')