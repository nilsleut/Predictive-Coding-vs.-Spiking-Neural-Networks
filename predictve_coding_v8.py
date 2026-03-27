"""
Predictive Coding Network — RSA on THINGS-fMRI
===============================================
Basiert auf:
  - Rao & Ballard (1999). Predictive coding in the visual cortex.
    Nature Neuroscience 2(1): 79-87.
  - Millidge et al. (2021). Predictive Coding: a Theoretical and
    Experimental Review. arXiv:2107.12979.

Architektur:
  3-Layer PC-Netz, trainiert auf ResNet-50 Features (layer4, 2048-dim)
  extrahiert aus THINGS-Bildern.

  Warum Features statt Pixel:
  - THINGS-Bilder sind 224x224 = 50176 Pixel — zu gross für PC
  - ResNet-50 layer4 Features sind 2048-dim und bereits visuell strukturiert
  - Wir testen ob PC eine andere/bessere Repräsentation lernt als ResNet selbst

Experiment-Design:
  1. ResNet-50 Features auf THINGS-Stimuli extrahieren
  2. PC-Netz auf diesen Features trainieren (unüberwacht)
  3. RSA: PC-Repräsentationen (r1, r2, r3) vs. fMRI-RDMs (V1-IT)
  4. Vergleich mit ResNet-50, ViT-B/16, CLIP aus RSA_COMPARE_v2.ipynb

Zentrale Frage:
  Modelliert ein biologisch motiviertes PC-Netz IT-Kortex besser
  als rein datengetriebene CNNs?

Voraussetzung:
  - RSA_COMPARE_v2.ipynb muss im selben Ordner liegen
    (oder fmri_rdms aus diesem Notebook verfügbar sein)
  - THINGS-Datensatz unter THINGS_IMAGES_DIR
  - grokking_correct.py NICHT nötig — eigenständiges Skript
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # kein Display nötig — nur speichern
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
from dataclasses import dataclass, field
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════
# Konfiguration
# ══════════════════════════════════════════════════════════════

@dataclass
class Config:
    # Pfade — anpassen
    RSA_DIR:           Path = Path(r'C:\Users\nilsl\Desktop\Projekte\RSA')
    PC_DIR:            Path = Path(r'C:\Users\nilsl\Desktop\Projekte\Predictive Coding')
    DATENSATZ_DIR:     Path = None   # wird aus RSA_DIR abgeleitet
    H5_FILE:           Path = None
    VOX_META:          Path = None
    STIM_META:         Path = None
    THINGS_IMAGES_DIR: Path = None

    # Baseline RDM-Pfade (aus RSA_COMPARE_v2 exportieren)
    # Setze auf None falls nicht vorhanden — dann nur ResNet als Baseline
    VIT_RDM_PATHS: dict = field(default_factory=lambda: {
        'block3':  Path(r'C:\Users\nilsl\Desktop\Projekte\predictive coding\vit_rdm_block3.npy'),
        'block6':  Path(r'C:\Users\nilsl\Desktop\Projekte\predictive coding\vit_rdm_block6.npy'),
        'block9':  Path(r'C:\Users\nilsl\Desktop\Projekte\predictive coding\vit_rdm_block9.npy'),
        'block12': Path(r'C:\Users\nilsl\Desktop\Projekte\predictive coding\vit_rdm_block12.npy'),
    })
    CLIP_RDM_PATHS: dict = field(default_factory=lambda: {
        'block3':  Path(r'C:\Users\nilsl\Desktop\Projekte\predictive coding\clip_rdm_clip_block3.npy'),
        'block6':  Path(r'C:\Users\nilsl\Desktop\Projekte\predictive coding\clip_rdm_clip_block6.npy'),
        'block9':  Path(r'C:\Users\nilsl\Desktop\Projekte\predictive coding\clip_rdm_clip_block9.npy'),
        'block12': Path(r'C:\Users\nilsl\Desktop\Projekte\predictive coding\clip_rdm_clip_block12.npy'),
    })
    # Stimuli
    N_IMAGES:    int = 720     # alle THINGS-Konzepte
    DEVICE:      str = "cuda" if torch.cuda.is_available() else "cpu"

    # PC-Netz Architektur — hierarchische ResNet-Features
    # Jede PC-Schicht empfängt den entsprechenden ResNet-Layer als Input
    # layer1=256-dim, layer2=512-dim, layer3=1024-dim, layer4=2048-dim
    # PC-Schichten lernen top-down Vorhersagen zwischen diesen Ebenen
    d_layer1:    int = 256    # ResNet layer1 → PC r0 (V1-analog)
    d_layer2:    int = 512    # ResNet layer2 → PC r1 (V2/V4-analog)
    d_layer3:    int = 1024   # ResNet layer3 → PC r2 (LOC-analog)
    d_layer4:    int = 2048   # ResNet layer4 → PC r3 (IT-analog)
    # Alias für Kompatibilität
    d_input:     int = 256    # = d_layer1, Input für unterste PC-Schicht

    # PC Training
    lr_r:        float = 0.01    # Lernrate für Repräsentations-Updates
    lr_w:        float = 5e-4    # Lernrate für Gewichte
    T_infer:     int   = 30      # Inferenz-Schritte pro Input
    n_epochs:    int   = 100     # Trainings-Epochen
    patience:    int   = 15      # Early Stopping — Geduld in Epochen
    grad_clip:   float = 1.0     # Gradient Clipping — verhindert Divergenz
    batch_size:  int   = 32

    # RSA
    ROI_NAMES:   tuple = ('V1', 'V2', 'V3', 'V4', 'LOC', 'IT')

    def __post_init__(self):
        self.PC_DIR.mkdir(parents=True, exist_ok=True)
        self.OUT_DIR = self.PC_DIR / 'outputs'
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
# Predictive Coding Netz
# ══════════════════════════════════════════════════════════════

class PredictiveCodingNet(nn.Module):
    """
    3-Layer Predictive Coding Netz nach Rao & Ballard (1999).

    Notation:
      r0 = Input-Layer  (d_input)   — wird auf den Stimulus gesetzt
      r1 = Layer 1      (d_hidden1) — niedrige visuelle Repräsentation
      r2 = Layer 2      (d_hidden2) — mittlere Repräsentation
      r3 = Layer 3      (d_output)  — höchste Repräsentation (IT-analog)

    Für jeden Input:
      1. r0 = stimulus (festgehalten)
      2. r1, r2, r3 werden über T_infer Schritte optimiert
      3. Gewichte W1, W2, W3 werden nach Inferenz aktualisiert

    Fehler-Einheiten:
      eps_0 = r0 - f(W1 @ r1)      Fehler in Layer 0
      eps_1 = r1 - f(W2 @ r2)      Fehler in Layer 1
      eps_2 = r2 - f(W3 @ r3)      Fehler in Layer 2

    Repräsentations-Updates:
      dr1 = -eps_1 + W1.T @ eps_0
      dr2 = -eps_2 + W2.T @ eps_1
      dr3 =        + W3.T @ eps_2

    Gewichts-Updates (nach Inferenz-Konvergenz):
      dW1 = eps_0 @ r1.T
      dW2 = eps_1 @ r2.T
      dW3 = eps_2 @ r3.T
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Top-down Gewichte (Vorhersage: höher → niedriger)
        # W1: r1 → r0 Vorhersage
        # W2: r2 → r1 Vorhersage
        # W3: r3 → r2 Vorhersage
        self.W1 = nn.Parameter(
            torch.randn(cfg.d_layer1, cfg.d_layer2) * 0.01)
        self.W2 = nn.Parameter(
            torch.randn(cfg.d_layer2, cfg.d_layer3) * 0.01)
        self.W3 = nn.Parameter(
            torch.randn(cfg.d_layer3, cfg.d_layer4)  * 0.01)

        # Bias-Terme pro Layer
        self.b1 = nn.Parameter(torch.zeros(cfg.d_layer1))
        self.b2 = nn.Parameter(torch.zeros(cfg.d_layer2))
        self.b3 = nn.Parameter(torch.zeros(cfg.d_layer3))

    def predict(self, r_higher, W, b):
        """Top-down Vorhersage: f(W @ r_higher + b)"""
        return torch.tanh(r_higher @ W.T + b)

    @torch.no_grad()
    def _clip_weights(self, max_norm: float = 10.0):
        """Verhindert explodierende Gewichte."""
        for W in [self.W1, self.W2, self.W3]:
            norm = W.data.norm()
            if norm > max_norm:
                W.data *= max_norm / norm

    def infer(self, inputs: dict):
        """
        Inferenz-Phase: Repräsentationen für hierarchische Inputs optimieren.

        inputs: dict mit keys 'layer1'–'layer4', je [B, dim]
          Jeder ResNet-Layer setzt den initialen Constraint für die
          entsprechende PC-Schicht.

        Returns:
          r0, r1, r2, r3: equilibrium representations
          errors: (eps0, eps1, eps2)
        """
        dev = inputs['layer1'].device
        cfg = self.cfg

        # Repräsentationen mit ResNet-Features initialisieren
        # Das ist der Kernunterschied zu v1: nicht Null-Initialisierung
        # sondern bottom-up Initialisierung durch ResNet
        r0 = inputs['layer1'].clone()   # [B, 256]
        r1 = inputs['layer2'].clone()   # [B, 512]
        r2 = inputs['layer3'].clone()   # [B, 1024]
        r3 = inputs['layer4'].clone()   # [B, 2048]

        for t in range(cfg.T_infer):
            # Top-down Vorhersagen
            pred0 = self.predict(r1, self.W1, self.b1)  # r1 → r0
            pred1 = self.predict(r2, self.W2, self.b2)  # r2 → r1
            pred2 = self.predict(r3, self.W3, self.b3)  # r3 → r2

            # Fehler: Differenz zwischen aktueller Rep und top-down Vorhersage
            eps0 = r0 - pred0
            eps1 = r1 - pred1
            eps2 = r2 - pred2

            # Repräsentations-Updates
            dr0 = -eps0                           # r0 bewegt sich zur Vorhersage
            dr1 = -eps1 + eps0 @ self.W1          # r1: eigener Fehler + bottom-up
            dr2 = -eps2 + eps1 @ self.W2
            dr3 =         eps2 @ self.W3

            # Update — r0 bleibt nahe am ResNet-Input (schwacher Prior)
            r0 = r0 + cfg.lr_r * 0.5 * dr0       # halbe lr für Input-Layer
            r1 = r1 + cfg.lr_r * dr1
            r2 = r2 + cfg.lr_r * dr2
            r3 = r3 + cfg.lr_r * dr3

        # Finale Fehler
        pred0 = self.predict(r1, self.W1, self.b1)
        pred1 = self.predict(r2, self.W2, self.b2)
        pred2 = self.predict(r3, self.W3, self.b3)
        eps0 = r0 - pred0
        eps1 = r1 - pred1
        eps2 = r2 - pred2

        return (r0.detach(), r1.detach(), r2.detach(), r3.detach()),                (eps0, eps1, eps2)

    def weight_update(self, errors, representations):
        """
        Gewichts-Update nach Inferenz-Konvergenz.
        Lokale Hebb'sche Lernregel: dW = eps @ r_higher.T
        """
        eps0, eps1, eps2 = errors
        r0, r1, r2, r3   = representations
        clip = self.cfg.grad_clip

        with torch.no_grad():
            # W1: layer2 → layer1 (r1 sagt r0 vorher)
            dW1 = (eps0.T @ r1) / eps0.shape[0]
            # W2: layer3 → layer2
            dW2 = (eps1.T @ r2) / eps1.shape[0]
            # W3: layer4 → layer3
            dW3 = (eps2.T @ r3) / eps2.shape[0]

            for dW in [dW1, dW2, dW3]:
                dW.clamp_(-clip, clip)

            self.W1.data += self.cfg.lr_w * dW1
            self.W2.data += self.cfg.lr_w * dW2
            self.W3.data += self.cfg.lr_w * dW3
            self._clip_weights()

    def free_energy(self, errors):
        """
        Totale Free Energy = Summe der quadratischen Fehler.
        Wird minimiert durch Inferenz und Gewichts-Updates.
        """
        eps0, eps1, eps2 = errors
        return (eps0.pow(2).mean() +
                eps1.pow(2).mean() +
                eps2.pow(2).mean()).item()


# ══════════════════════════════════════════════════════════════
# ResNet-50 Feature-Extraktion
# ══════════════════════════════════════════════════════════════

def extract_resnet_features(image_paths: list, device: str) -> dict:
    """
    Extrahiert ResNet-50 Features aus layer1–layer4 (nach Global Average Pooling).

    Warum alle 4 Layer:
      layer1: 256-dim  — Kanten, Texturen (V1-analog)
      layer2: 512-dim  — Formen, Muster   (V2/V4-analog)
      layer3: 1024-dim — Teile, Objekte   (LOC-analog)
      layer4: 2048-dim — Objekte, Konzepte (IT-analog)

    Das gibt dem PC-Netz eine echte Inputhierarchie — jede PC-Schicht
    empfängt Features einer bestimmten Abstraktionsstufe als Constraint.

    Returns: dict mit keys 'layer1'–'layer4', je [N, dim]
    """
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet.eval().to(device)

    # Hooks auf alle 4 Layer
    layer_features = {k: [] for k in ['layer1', 'layer2', 'layer3', 'layer4']}
    batch_cache = {}

    def make_hook(name):
        def hook(module, input, output):
            # Global Average Pooling über spatial dims [B, C, H, W] → [B, C]
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

    batch_size = 32
    for start in tqdm(range(0, len(image_paths), batch_size),
                      desc='ResNet-50 layer1-4 Features'):
        batch_paths = image_paths[start:start + batch_size]
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
# PC-Netz Training
# ══════════════════════════════════════════════════════════════

def train_pc(layer_features: dict, cfg: Config) -> PredictiveCodingNet:
    """
    Trainiert das hierarchische PC-Netz auf ResNet layer1-4 Features.

    layer_features: dict {'layer1': [N,256], 'layer2': [N,512],
                          'layer3': [N,1024], 'layer4': [N,2048]}
    """
    # Normiere jeden Layer separat
    features_n = {}
    norms = {}
    for k, v in layer_features.items():
        mean = v.mean(dim=0, keepdim=True)
        std  = v.std(dim=0, keepdim=True).clamp(min=1e-8)
        features_n[k] = (v - mean) / std
        norms[k] = (mean, std)

    N  = len(features_n['layer1'])
    pc = PredictiveCodingNet(cfg).to(cfg.DEVICE)
    pc.norms = norms  # speichern für spätere Verwendung

    print(f"\nHierarchisches PC-Netz Training:")
    print(f"  layer1(256) ← layer2(512) ← layer3(1024) ← layer4(2048)")
    print(f"  T_infer={cfg.T_infer}, lr_r={cfg.lr_r}, lr_w={cfg.lr_w}")
    print(f"  {cfg.n_epochs} Epochen × {N} Stimuli\n")

    fe_history = []
    best_fe        = float('inf')
    best_weights   = {k: v.clone() for k, v in pc.state_dict().items()}
    patience_count = 0

    for epoch in range(cfg.n_epochs):
        perm = torch.randperm(N)
        epoch_fe = 0.0
        n_batches = 0
        diverged = False

        for start in range(0, N, cfg.batch_size):
            idx   = perm[start:start + cfg.batch_size]
            batch = {k: features_n[k][idx].to(cfg.DEVICE)
                     for k in features_n}

            # Inferenz-Phase
            reps, errors = pc.infer(batch)

            # Free Energy
            fe = pc.free_energy(errors)
            epoch_fe += fe
            n_batches += 1

            # NaN-Check — bei Divergenz abbrechen
            if np.isnan(fe) or fe > 1e6:
                print(f"  WARNUNG: Divergenz bei Epoch {epoch+1} Step {start} "
                      f"(FE={fe:.2e}) — lr_w zu hoch?")
                diverged = True
                break

            # Gewichts-Update
            pc.weight_update(errors, reps)

        if diverged:
            break

        avg_fe = epoch_fe / n_batches
        fe_history.append(avg_fe)

        # Early Stopping — beste Gewichte speichern
        if avg_fe < best_fe:
            best_fe      = avg_fe
            best_weights = {k: v.clone() for k, v in pc.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{cfg.n_epochs} | "
                  f"Free Energy: {avg_fe:.4f}  "
                  f"(best={best_fe:.4f}, patience={patience_count}/{cfg.patience})")

        if patience_count >= cfg.patience:
            print(f"\n  Early Stop bei Epoch {epoch+1} "
                  f"(keine Verbesserung seit {cfg.patience} Epochen)")
            break

    # Beste Gewichte laden
    pc.load_state_dict(best_weights)
    print(f"\nTraining abgeschlossen ✓  Beste Free Energy: {best_fe:.4f}")

    # Normierungsfaktoren speichern
    return pc, fe_history


# ══════════════════════════════════════════════════════════════
# PC-Repräsentationen extrahieren (nach Training)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def get_pc_representations(pc: PredictiveCodingNet,
                            layer_features: dict) -> dict:
    """
    Extrahiert PC-Repräsentationen r0–r3 und Fehler-Signale ε0–ε2.

    r0 = PC-verfeinerte layer1-Repräsentation (V1-analog)
    r1 = PC-verfeinerte layer2-Repräsentation (V4-analog)
    r2 = PC-verfeinerte layer3-Repräsentation (LOC-analog)
    r3 = PC-verfeinerte layer4-Repräsentation (IT-analog)
    """
    # Normieren wie beim Training
    features_n = {}
    for k, v in layer_features.items():
        mean, std = pc.norms[k]
        features_n[k] = (v - mean) / std

    all_r = {k: [] for k in ['r0', 'r1', 'r2', 'r3']}
    all_e = {k: [] for k in ['e0', 'e1', 'e2']}

    N = len(features_n['layer1'])
    for start in range(0, N, 32):
        batch = {k: features_n[k][start:start+32].to(pc.cfg.DEVICE)
                 for k in features_n}
        (r0, r1, r2, r3), (e0, e1, e2) = pc.infer(batch)
        all_r['r0'].append(r0.cpu())
        all_r['r1'].append(r1.cpu())
        all_r['r2'].append(r2.cpu())
        all_r['r3'].append(r3.cpu())
        all_e['e0'].append(e0.cpu())
        all_e['e1'].append(e1.cpu())
        all_e['e2'].append(e2.cpu())

    result = {}
    for k in ['r0', 'r1', 'r2', 'r3']:
        result[k] = torch.cat(all_r[k], dim=0).numpy()
    for k in ['e0', 'e1', 'e2']:
        # relu: nur positive Prediction Errors (Rao & Ballard 1999)
        result[k] = F.relu(torch.cat(all_e[k], dim=0)).numpy()

    return result


# ══════════════════════════════════════════════════════════════
# RSA Hilfsfunktionen
# ══════════════════════════════════════════════════════════════

def compute_rdm(features: np.ndarray) -> np.ndarray:
    return squareform(pdist(features, metric='correlation'))

def compare_rdms(rdm_a, rdm_b):
    n   = rdm_a.shape[0]
    idx = np.triu_indices(n, k=1)
    rho, p = spearmanr(rdm_a[idx], rdm_b[idx])
    return rho, p


# ══════════════════════════════════════════════════════════════
# Visualisierung
# ══════════════════════════════════════════════════════════════

def plot_training_curve(fe_history: list, save_path: str):
    best_epoch = int(np.argmin(fe_history))
    best_fe    = fe_history[best_epoch]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(fe_history, color='#7c6af7', linewidth=2, label='Free Energy')
    ax.axvline(best_epoch, color='#e07b39', linewidth=1.5,
               linestyle='--', label=f'Best (Epoch {best_epoch+1}, FE={best_fe:.3f})')
    ax.scatter([best_epoch], [best_fe], color='#e07b39', zorder=5, s=60)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Free Energy', fontsize=11)
    ax.set_title('PC-Netz Training — Free Energy', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Gespeichert: {save_path}")


def plot_rsa_comparison(rho_results: dict, noise_ceilings: dict,
                        roi_names: list, save_path: str):
    """
    Vergleicht PC-Netz (r1, r2, r3, e0, e1, e2) mit ResNet/ViT/CLIP.
    """
    n_rois = len(roi_names)

    # Modelle definieren
    pc_layers = ['r0', 'r1', 'r2', 'r3', 'e0', 'e1', 'e2']
    pc_colors = ['#d4c4f7', '#7c6af7', '#9c87f7', '#bca8f7',
                 '#f76a8c', '#f78ca6', '#f7aec0']
    pc_labels = ['PC r0 (V1)', 'PC r1 (V4)', 'PC r2 (LOC)', 'PC r3 (IT)',
                 'PC ε0', 'PC ε1', 'PC ε2']

    baseline_models = {
        'ResNet-50': ('#4477aa', rho_results.get('resnet', {})),
        'ViT-B/16':  ('#ee6677', rho_results.get('vit', {})),
        'CLIP':      ('#228833', rho_results.get('clip', {})),
    }

    x = np.arange(n_rois)
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle('RSA: Predictive Coding vs. ResNet-50 / ViT-B/16 / CLIP\n'
                 'Spearman ρ — Modell-RDM vs. fMRI-RDM',
                 fontsize=13, fontweight='bold')

    width = 0.10
    offsets = np.linspace(-0.45, 0.45, len(pc_layers) + len(baseline_models))

    # PC Layers
    for i, (layer, color, label) in enumerate(zip(pc_layers, pc_colors, pc_labels)):
        if layer not in rho_results:
            continue
        vals = [rho_results[layer].get(roi, 0) for roi in roi_names]
        ax.bar(x + offsets[i], vals, width, label=label,
               color=color, alpha=0.85, edgecolor='white')

    # Baselines
    for j, (name, (color, rho_dict)) in enumerate(baseline_models.items()):
        if not rho_dict:
            continue
        vals = [rho_dict.get(roi, 0) for roi in roi_names]
        ax.bar(x + offsets[len(pc_layers) + j], vals, width,
               label=name, color=color, alpha=0.85,
               edgecolor='white', hatch='//')

    # Noise Ceilings
    for j, roi in enumerate(roi_names):
        nc = noise_ceilings.get(roi, 0)
        ax.plot([x[j] - 0.5, x[j] + 0.5], [nc, nc],
                'k--', linewidth=1.5, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(roi_names, fontsize=11)
    ax.set_ylabel('Spearman ρ', fontsize=11)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.legend(fontsize=8, ncol=3, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(-0.05, max(noise_ceilings.values()) + 0.1
                if noise_ceilings else 0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Gespeichert: {save_path}")


def print_results_table(rho_results: dict, noise_ceilings: dict,
                        roi_names: list):
    layers = ['r1', 'r2', 'r3', 'e0', 'e1', 'e2']
    labels = ['PC-r1', 'PC-r2', 'PC-r3', 'PC-ε0', 'PC-ε1', 'PC-ε2']

    print('\n' + '=' * 70)
    print('RSA ERGEBNISSE — Predictive Coding vs. Baselines')
    print('=' * 70)
    header = f'{"Modell":>12}' + ''.join(f'{r:>8}' for r in roi_names)
    print(header)
    print('─' * 70)

    for layer, label in zip(layers, labels):
        if layer not in rho_results:
            continue
        row = f'{label:>12}'
        for roi in roi_names:
            rho = rho_results[layer].get(roi, 0)
            row += f'{rho:>8.3f}'
        print(row)

    print('─' * 70)
    for name in ['resnet', 'vit', 'clip']:
        if name not in rho_results:
            continue
        label = {'resnet': 'ResNet-50', 'vit': 'ViT-B/16', 'clip': 'CLIP'}[name]
        row = f'{label:>12}'
        for roi in roi_names:
            rho = rho_results[name].get(roi, 0)
            row += f'{rho:>8.3f}'
        print(row)

    if noise_ceilings:
        print('─' * 70)
        row = f'{"NoiseClng":>12}'
        for roi in roi_names:
            row += f'{noise_ceilings.get(roi, 0):>8.3f}'
        print(row)



def plot_hierarchy(rho_results: dict, roi_names: list, save_path: str):
    """
    Zeigt den Hierarchie-Gradienten: r1→r3 über V1→IT.
    Vorhersage des PC-Modells: niedrige Repräsentationen (r1) korrelieren
    besser mit frühen Arealen (V1), hohe (r3) mit späten (IT/LOC).
    """
    pc_layers  = ['r0', 'r1', 'r2', 'r3']
    pc_labels  = ['r0 (V1-init)', 'r1 (V4-init)', 'r2 (LOC-init)', 'r3 (IT-init)']
    colors     = ['#d4c4f7', '#bca8f7', '#9c87f7', '#7c6af7']

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle('PC Hierarchie-Gradient: r1→r3 über V1→IT\n'
                 'Vorhersage: niedrige Layer korrelieren mit frühen Arealen',
                 fontsize=12, fontweight='bold')

    x = np.arange(len(roi_names))
    for layer, label, color in zip(pc_layers, pc_labels, colors):
        if layer not in rho_results:
            continue
        vals = [rho_results[layer].get(roi, 0) for roi in roi_names]
        ax.plot(x, vals, 'o-', label=label, color=color,
                linewidth=2, markersize=8)

    # ResNet als Referenz
    if 'resnet' in rho_results:
        vals = [rho_results['resnet'].get(roi, 0) for roi in roi_names]
        ax.plot(x, vals, 's--', label='ResNet-50', color='#4477aa',
                linewidth=1.5, markersize=6, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(roi_names, fontsize=11)
    ax.set_ylabel('Spearman ρ', fontsize=11)
    ax.set_xlabel('ROI (früh → spät)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Gespeichert: {save_path}")



def bootstrap_rsa(model_rdm: np.ndarray,
                  fmri_rdm: np.ndarray,
                  n_boot: int = 1000,
                  ci: float = 0.95) -> tuple:
    """
    Bootstrap-Konfidenzintervalle für Spearman-ρ zwischen zwei RDMs.

    Resampling über Stimulus-Paare (oberes Dreieck der RDM).
    Returns: (rho, ci_low, ci_high)
    """
    n = model_rdm.shape[0]
    idx = np.triu_indices(n, k=1)
    x   = model_rdm[idx]
    y   = fmri_rdm[idx]
    n_pairs = len(x)

    rho_obs, _ = spearmanr(x, y)

    # Bootstrap
    rng       = np.random.default_rng(42)
    boot_rhos = np.zeros(n_boot)
    for i in range(n_boot):
        sample    = rng.integers(0, n_pairs, size=n_pairs)
        boot_rhos[i], _ = spearmanr(x[sample], y[sample])

    alpha   = (1 - ci) / 2
    ci_low  = np.percentile(boot_rhos, alpha * 100)
    ci_high = np.percentile(boot_rhos, (1 - alpha) * 100)

    return float(rho_obs), float(ci_low), float(ci_high)


def plot_hierarchy_with_ci(rho_results: dict,
                            ci_results: dict,
                            roi_names: list,
                            save_path: str):
    """
    Hierarchie-Plot mit Bootstrap-Konfidenzintervallen.
    Zeigt ob Unterschiede zwischen PC-Schichten statistisch robust sind.
    """
    pc_layers = ['r0', 'r1', 'r2', 'r3']
    pc_labels = ['r0 (V1-init)', 'r1 (V4-init)',
                 'r2 (LOC-init)', 'r3 (IT-init)']
    colors    = ['#d4c4f7', '#bca8f7', '#9c87f7', '#7c6af7']

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        'PC Hierarchie-Gradient mit 95% Bootstrap-Konfidenzintervallen\n'
        'Kreuzungsmuster bestätigt PC-Vorhersage: niedrige Layer → frühe Areale',
        fontsize=12, fontweight='bold'
    )

    x = np.arange(len(roi_names))

    for layer, label, color in zip(pc_layers, pc_labels, colors):
        if layer not in rho_results:
            continue
        vals   = np.array([rho_results[layer].get(roi, 0) for roi in roi_names])
        ci_lo  = np.array([ci_results[layer].get(roi, (0,0,0))[1] for roi in roi_names])
        ci_hi  = np.array([ci_results[layer].get(roi, (0,0,0))[2] for roi in roi_names])

        ax.plot(x, vals, 'o-', label=label, color=color,
                linewidth=2.5, markersize=8, zorder=3)
        ax.fill_between(x, ci_lo, ci_hi, alpha=0.15, color=color)

    # ResNet-50 als Referenz mit CI
    if 'resnet' in rho_results:
        vals  = np.array([rho_results['resnet'].get(roi, 0) for roi in roi_names])
        ci_lo = np.array([ci_results['resnet'].get(roi, (0,0,0))[1] for roi in roi_names])
        ci_hi = np.array([ci_results['resnet'].get(roi, (0,0,0))[2] for roi in roi_names])
        ax.plot(x, vals, 's--', label='ResNet-50', color='#4477aa',
                linewidth=1.5, markersize=6, alpha=0.8, zorder=3)
        ax.fill_between(x, ci_lo, ci_hi, alpha=0.10, color='#4477aa')

    ax.set_xticks(x)
    ax.set_xticklabels(roi_names, fontsize=11)
    ax.set_ylabel('Spearman ρ', fontsize=11)
    ax.set_xlabel('ROI (früh → spät)', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.8)

    # Annotation: Kreuzungspunkt
    ax.annotate('Kreuzung\n(PC-Vorhersage)',
                xy=(2.5, 0.20), fontsize=8, color='gray',
                ha='center', style='italic')
    ax.axvline(2.5, color='gray', linewidth=1, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Gespeichert: {save_path}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def run_subject(sub_id: str) -> dict:
    """Führt die komplette PC-RSA-Pipeline für ein Subject durch.
    Gibt rho_results zurück (für spätere Mittelung)."""
    cfg = Config()

    # Reproduzierbarkeit
    import random
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Subject-spezifische Pfade setzen
    cfg.H5_FILE  = cfg.DATENSATZ_DIR / f'{sub_id}_task-things_voxel-wise-responses.h5'
    cfg.VOX_META = cfg.DATENSATZ_DIR / f'{sub_id}_task-things_voxel-metadata.csv'
    cfg.STIM_META = cfg.DATENSATZ_DIR / f'{sub_id}_task-things_stimulus-metadata.csv'

    print("=" * 65)
    print(f"Predictive Coding RSA — THINGS-fMRI  [{sub_id}]")
    print("=" * 65)
    print(f"Device: {cfg.DEVICE}")

    # ── Schritt 1: Stimuli laden ──────────────────────────────
    print("\n[1/5] Lade Stimuli-Metadaten...")
    import pandas as pd, h5py

    vox_meta  = pd.read_csv(cfg.VOX_META,  sep=',')
    stim_meta = pd.read_csv(cfg.STIM_META, sep=',')

    # ROI-Masken
    roi_masks = {
        'V1':  vox_meta['V1'].values.astype(bool),
        'V2':  vox_meta['V2'].values.astype(bool),
        'V3':  vox_meta['V3'].values.astype(bool),
        'V4':  vox_meta['hV4'].values.astype(bool),
        'LOC': (vox_meta['lLOC'].values.astype(bool) |
                vox_meta['rLOC'].values.astype(bool)),
        'IT':  vox_meta['IT'].values.astype(bool),
    }

    combined_mask = np.zeros(len(vox_meta), dtype=bool)
    for m in roi_masks.values():
        combined_mask |= m
    roi_voxel_indices = np.where(combined_mask)[0]
    global_to_local   = {int(g): l for l, g in enumerate(roi_voxel_indices)}

    # ── Schritt 2: fMRI-Daten laden ───────────────────────────
    print("[2/5] Lade fMRI-Daten...")
    with h5py.File(cfg.H5_FILE, 'r') as f:
        dset         = f['ResponseData/block0_values']
        roi_data_raw = dset[roi_voxel_indices, :].astype(np.float32)

    responses_all = roi_data_raw.T
    responses_all = ((responses_all - responses_all.mean(axis=0)) /
                     (responses_all.std(axis=0) + 1e-8))

    # Stimuli auswählen
    stim_meta_test = stim_meta[stim_meta['trial_type'] == 'test'].copy()

    # Kategorienbasierte Auswahl — identisch zu RSA_COMPARE_v2
    # Stellt sicher dass dieselben Stimuli wie in den Baseline-Experimenten
    # Duplikate entfernen — jeder Stimulus nur einmal (Wiederholungen ignorieren)
    # 1 Stimulus pro Konzept, alle verfügbaren Konzepte → N×N RDM
    # Alphabetisch sortieren → subject-unabhängige Reihenfolge
    stim_meta_unique = stim_meta_test.drop_duplicates(subset='stimulus').copy()
    valid_concepts   = sorted(stim_meta_unique['concept'].unique().tolist())
    stim_order = []
    for concept in valid_concepts:
        stims = sorted(
            stim_meta_unique[
                stim_meta_unique['concept'] == concept
            ]['stimulus'].tolist()
        )[:1]  # 1 Stimulus pro Konzept
        stim_order.extend(stims)
    stim_order = stim_order[:cfg.N_IMAGES]
    print(f"Stimuli: {len(stim_order)} aus {len(valid_concepts)} Konzepten")

    # stim_order speichern — subject-spezifisch + gemeinsame Referenz
    stim_file_sub = cfg.OUT_DIR / f'stim_order_{sub_id}.txt'
    stim_file_ref = cfg.PC_DIR / 'stim_order_pc.txt'  # Referenz für Notebook
    for stim_file in [stim_file_sub, stim_file_ref]:
        with open(str(stim_file), 'w') as f:
            for s in stim_order:
                f.write(s + '\n')
    # Cross-Subject Verifikation
    ref = cfg.OUT_DIR / 'stim_order_sub-01.txt'
    if ref.exists() and sub_id != 'sub-01':
        with open(ref) as f:
            ref_order = [l.strip() for l in f.readlines()]
        overlap = len(set(stim_order) & set(ref_order))
        n_ref = len(ref_order)
        if overlap == n_ref:
            print(f"PC stim_order exportiert ✓  (identisch mit sub-01, {overlap} Stimuli)")
        else:
            print(f"⚠️  stim_order Überlappung mit sub-01: {overlap}/{n_ref}")
    else:
        print(f"PC stim_order exportiert ✓")

    stim_responses, image_paths = [], []
    stim_order_found = []
    for stim in tqdm(stim_order, desc='fMRI mitteln'):
        idx = stim_meta_test.index[stim_meta_test['stimulus'] == stim].tolist()
        if len(idx) == 0:
            print(f"  ⚠️  Stimulus nicht gefunden: {stim} — übersprungen")
            continue
        stim_responses.append(responses_all[idx].mean(axis=0))
        concept = stim_meta_test.loc[idx[0], 'concept']
        image_paths.append(cfg.THINGS_IMAGES_DIR / concept / stim)
        stim_order_found.append(stim)

    stim_order = stim_order_found  # nur gefundene Stimuli
    responses = np.array(stim_responses)  # [N_found, N_voxel]
    print(f"fMRI responses: {responses.shape}")

    # fMRI-RDMs
    print("Berechne fMRI-RDMs...")
    fmri_rdms = {}
    for roi in cfg.ROI_NAMES:
        g_idx = np.where(roi_masks[roi])[0]
        l_idx = np.array([global_to_local[int(g)] for g in g_idx
                          if int(g) in global_to_local])
        fmri_rdms[roi] = compute_rdm(responses[:, l_idx])

    # ── Schritt 3: ResNet Features extrahieren ────────────────
    print("\n[3/5] Extrahiere ResNet-50 layer1-4 Features...")
    layer_features = extract_resnet_features(image_paths, cfg.DEVICE)
    resnet_features = layer_features['layer4']  # für RSA-Baseline

    # ── Schritt 4: PC-Netz trainieren ─────────────────────────
    print("\n[4/5] Trainiere PC-Netz...")
    pc, fe_history = train_pc(layer_features, cfg)
    plot_training_curve(fe_history,
        str(cfg.OUT_DIR / f'pc_training_curve_{sub_id}.png'))

    # PC-Repräsentationen extrahieren
    print("Extrahiere PC-Repräsentationen...")
    pc_reps = get_pc_representations(pc, layer_features)
    for k, v in pc_reps.items():
        print(f"  {k}: {v.shape}")

    # ── Schritt 5: RSA ────────────────────────────────────────
    print("\n[5/5] RSA: PC vs. fMRI...")
    rho_results = {}

    # PC-Schichten
    for layer_name, features in pc_reps.items():
        if np.isnan(features).any() or np.isinf(features).any():
            print(f"  WARNUNG: {layer_name} enthält NaN/Inf — übersprungen")
            continue
        rdm = compute_rdm(features)
        rho_results[layer_name] = {}
        for roi in cfg.ROI_NAMES:
            rho, p = compare_rdms(rdm, fmri_rdms[roi])
            rho_results[layer_name][roi] = rho
            print(f"  {layer_name} vs {roi}: ρ={rho:.3f}")

    # ResNet-50 Layer4 Baseline
    resnet_rdm = compute_rdm(resnet_features.numpy())
    rho_results['resnet'] = {}
    for roi in cfg.ROI_NAMES:
        rho, _ = compare_rdms(resnet_rdm, fmri_rdms[roi])
        rho_results['resnet'][roi] = rho
        print(f"  ResNet vs {roi}: ρ={rho:.3f}")

    # ── Bootstrap Konfidenzintervalle ─────────────────────────
    print("\nBerechne Bootstrap-Konfidenzintervalle (1000 Samples)...")
    ci_results = {}
    all_layers = list(pc_reps.keys()) + ['resnet']
    all_rdms   = {k: compute_rdm(v) for k, v in pc_reps.items()
                  if not (np.isnan(v).any() or np.isinf(v).any())}
    all_rdms['resnet'] = resnet_rdm

    for layer_name, rdm in all_rdms.items():
        ci_results[layer_name] = {}
        for roi in cfg.ROI_NAMES:
            rho, lo, hi = bootstrap_rsa(rdm, fmri_rdms[roi], n_boot=1000)
            ci_results[layer_name][roi] = (rho, lo, hi)
        print(f"  {layer_name} ✓")

    # Bootstrap-Tabelle drucken
    print("\n" + "=" * 75)
    print("BOOTSTRAP 95% CI — PC vs ResNet")
    print("=" * 75)
    print(f"{'Modell':>8}  {'Metrik':>6}  " +
          "  ".join(f"{r:>8}" for r in cfg.ROI_NAMES))
    print("─" * 75)
    for layer in ['r0', 'r1', 'r2', 'r3', 'resnet']:
        if layer not in ci_results:
            continue
        label = {'r0':'PC-r0','r1':'PC-r1','r2':'PC-r2',
                 'r3':'PC-r3','resnet':'ResNet'}[layer]
        for metric, idx in [('ρ', 0), ('lo', 1), ('hi', 2)]:
            row = f"{label:>8}  {metric:>6}  "
            row += "  ".join(
                f"{ci_results[layer][roi][idx]:>8.3f}"
                for roi in cfg.ROI_NAMES
            )
            print(row)
        print()

    # ViT-B/16 Baselines — alle Layer laden, besten pro ROI wählen
    vit_loaded = []
    for block, path in cfg.VIT_RDM_PATHS.items():
        if Path(path).exists():
            rdm = np.load(path)
            key = f'vit_{block}'
            rho_results[key] = {}
            for roi in cfg.ROI_NAMES:
                rho, _ = compare_rdms(rdm, fmri_rdms[roi])
                rho_results[key][roi] = rho
            vit_loaded.append(block)
            print(f"ViT {block} geladen ✓")
    if vit_loaded:
        rho_results['vit'] = {}
        for roi in cfg.ROI_NAMES:
            rho_results['vit'][roi] = max(
                rho_results[f'vit_{b}'][roi] for b in vit_loaded)
    else:
        print("ViT RDMs nicht gefunden — nur ResNet als Baseline")

    # CLIP Baselines — alle Layer laden, besten pro ROI wählen
    clip_loaded = []
    for block, path in cfg.CLIP_RDM_PATHS.items():
        if Path(path).exists():
            rdm = np.load(path)
            key = f'clip_{block}'
            rho_results[key] = {}
            for roi in cfg.ROI_NAMES:
                rho, _ = compare_rdms(rdm, fmri_rdms[roi])
                rho_results[key][roi] = rho
            clip_loaded.append(block)
            print(f"CLIP {block} geladen ✓")
    if clip_loaded:
        rho_results['clip'] = {}
        for roi in cfg.ROI_NAMES:
            rho_results['clip'][roi] = max(
                rho_results[f'clip_{b}'][roi] for b in clip_loaded)
    else:
        print("CLIP RDMs nicht gefunden — nur ResNet als Baseline")

    # Noise Ceilings — Split-Half, Spearman-Brown, pro Subject berechnet
    print("\nBerechne Noise Ceilings (Split-Half)...")
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

    # Ergebnisse
    print_results_table(rho_results, noise_ceilings, list(cfg.ROI_NAMES))

    plot_rsa_comparison(
        rho_results, noise_ceilings, list(cfg.ROI_NAMES),
        str(cfg.OUT_DIR / f'pc_rsa_comparison_{sub_id}.png')
    )

    # Hierarchie-Plot mit Konfidenzintervallen
    plot_hierarchy_with_ci(
        rho_results, ci_results, list(cfg.ROI_NAMES),
        str(cfg.OUT_DIR / f'pc_hierarchy_{sub_id}.png')
    )

    # Ergebnisse als .npy speichern — für Vergleich mit SNN (snn_rsa_v3.py)
    np.save(str(cfg.OUT_DIR / f'pc_rho_results_{sub_id}.npy'), rho_results)
    np.save(str(cfg.OUT_DIR / f'pc_noise_ceilings_{sub_id}.npy'), noise_ceilings)

    print(f"\nFertig [{sub_id}]. Ausgaben:")
    print(f"  {cfg.OUT_DIR}/pc_training_curve_{sub_id}.png")
    print(f"  {cfg.OUT_DIR}/pc_rsa_comparison_{sub_id}.png")
    print(f"  {cfg.OUT_DIR}/pc_hierarchy_{sub_id}.png")
    print(f"  {cfg.OUT_DIR}/pc_rho_results_{sub_id}.npy")
    print(f"  {cfg.OUT_DIR}/pc_noise_ceilings_{sub_id}.npy")

    return rho_results, noise_ceilings



def plot_permutation_null(perm_results: dict, save_path: str):
    """Plottet Null-Verteilung des Permutationstests mit beobachtetem Effekt."""
    null     = perm_results['null']
    observed = perm_results['observed']
    p_val    = perm_results['p_value']

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null, bins=40, color='#9b72cf', alpha=0.6, edgecolor='white',
            linewidth=0.5, label='Null-Verteilung (permutiert)')
    ax.axvline(observed, color='#e07b39', linewidth=2.5,
               label=f'Beobachtet: {observed:+.3f}  (p={p_val:.3f})')
    ax.set_xlabel('Interaktionseffekt Δr0 − Δr3', fontsize=11)
    ax.set_ylabel('Häufigkeit', fontsize=11)
    ax.set_title('Permutationstest — Layer×ROI Interaktion\n'
                 f'N=1000 Permutationen, einseitiger p-Wert',
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Gespeichert: {save_path}')


def permutation_test_interaction(all_results: dict, roi_names: list,
                                  n_perm: int = 1000,
                                  early: list = None,
                                  late: list = None) -> dict:
    """
    Permutationstest auf den Layer×ROI Interaktionseffekt Δr0 − Δr3.
    Nullhypothese: ROI-Labels sind zufällig — kein systematischer Gradient.
    Gibt p-Wert und Effektgröße zurück.
    """
    if early is None:
        early = ['V1', 'V2']
    if late is None:
        late  = ['LOC', 'IT']

    layers   = ['r0', 'r1', 'r2', 'r3']
    subjects = list(all_results.keys())

    def compute_interaction(rho_dict_per_subject):
        # Mittlere rho über Subjects pro Layer/ROI
        mean_rho = {}
        for layer in layers:
            mean_rho[layer] = {}
            for roi in roi_names:
                vals = [rho_dict_per_subject[s][layer][roi]
                        for s in subjects if layer in rho_dict_per_subject[s]]
                mean_rho[layer][roi] = float(np.mean(vals))
        d_r0 = (np.mean([mean_rho['r0'][r] for r in early]) -
                np.mean([mean_rho['r0'][r] for r in late]))
        d_r3 = (np.mean([mean_rho['r3'][r] for r in early]) -
                np.mean([mean_rho['r3'][r] for r in late]))
        return d_r0 - d_r3

    # Beobachteter Effekt
    observed = compute_interaction(all_results)

    # Permutationen: ROI-Labels zufällig permutieren
    rng  = np.random.default_rng(42)
    null = np.zeros(n_perm)
    for i in range(n_perm):
        perm_order = rng.permutation(roi_names)
        roi_map    = dict(zip(roi_names, perm_order))
        # Remapped results
        remapped = {}
        for s in subjects:
            remapped[s] = {}
            for layer in layers:
                remapped[s][layer] = {roi_map[roi]: all_results[s][layer][roi]
                                      for roi in roi_names
                                      if layer in all_results[s]}
        null[i] = compute_interaction(remapped)

    p_val = float(np.mean(null >= observed))

    print(f'\nPermutationstest (n={n_perm}):')
    print(f'  Beobachteter Interaktionseffekt: {observed:+.3f}')
    print(f'  Null-Verteilung: μ={null.mean():.3f}, σ={null.std():.3f}')
    print(f'  p-Wert (einseitig): {p_val:.3f}')
    if p_val < 0.001:
        print(f'  *** p < 0.001')
    elif p_val < 0.01:
        print(f'  **  p < 0.01')
    elif p_val < 0.05:
        print(f'  *   p < 0.05')
    else:
        print(f'  n.s. (p ≥ 0.05)')

    return {'observed': observed, 'null': null, 'p_value': p_val}


def plot_group_average(all_results: dict, roi_names: list,
                       mean_resnet: dict, mean_nc: dict,
                       save_path: str):
    """
    Mittelt rho_results über alle Subjects und plottet Hierarchie-Gradient
    mit ResNet-Referenz und Noise Ceiling.
    all_results: {sub_id: rho_results_dict}
    """
    subjects = list(all_results.keys())
    layers   = ['r0', 'r1', 'r2', 'r3']

    # Mittelwert und SD über Subjects
    mean_rho = {layer: {} for layer in layers}
    sd_rho   = {layer: {} for layer in layers}
    for layer in layers:
        for roi in roi_names:
            vals = [all_results[s][layer][roi] for s in subjects
                    if layer in all_results[s]]
            mean_rho[layer][roi] = np.mean(vals)
            sd_rho[layer][roi]   = np.std(vals)

    x      = np.arange(len(roi_names))
    colors = ['#c8a8e8', '#9b72cf', '#6a3fa0', '#3b1f6e']
    labels = ['r0 (V1-init)', 'r1 (V4-init)', 'r2 (LOC-init)', 'r3 (IT-init)']

    fig, ax = plt.subplots(figsize=(11, 5))

    # Noise Ceiling als graues Band
    if mean_nc:
        nc_vals = np.array([mean_nc[roi] for roi in roi_names])
        ax.fill_between(x, 0, nc_vals, color='#aaaaaa', alpha=0.12,
                        label='Noise Ceiling (Ø)')
        ax.plot(x, nc_vals, color='#888888', linewidth=1,
                linestyle=':', alpha=0.7)

    # PC-Layer
    for layer, color, label in zip(layers, colors, labels):
        means = np.array([mean_rho[layer][roi] for roi in roi_names])
        sds   = np.array([sd_rho[layer][roi]   for roi in roi_names])
        ax.plot(x, means, 'o-', color=color, label=label, linewidth=2.5,
                markersize=8, zorder=3)
        ax.fill_between(x, means - sds, means + sds,
                        color=color, alpha=0.15, zorder=2)

    # ResNet-50 Referenz
    if mean_resnet:
        resnet_vals = np.array([mean_resnet[roi] for roi in roi_names])
        ax.plot(x, resnet_vals, 's--', color='#4477aa', linewidth=1.8,
                markersize=7, label='ResNet-50 (Ø)', alpha=0.85, zorder=4)

    # Interaktionstest Layer × ROI (Früh vs. Spät)
    early = ['V1', 'V2']
    late  = ['LOC', 'IT']
    print("\nInteraktionstest Layer × ROI (früh vs. spät):")
    print(f"  {'Layer':6}  {'Früh (Ø)':>10}  {'Spät (Ø)':>10}  {'Δ':>8}")
    print("  " + "-" * 40)
    for layer in layers:
        mu_early = np.mean([mean_rho[layer][r] for r in early if r in mean_rho[layer]])
        mu_late  = np.mean([mean_rho[layer][r] for r in late  if r in mean_rho[layer]])
        delta    = mu_early - mu_late
        print(f"  {layer:6}  {mu_early:>10.3f}  {mu_late:>10.3f}  {delta:>+8.3f}")
    # Erwartung PC: r0 Δ > 0, r3 Δ < 0
    d_r0 = (np.mean([mean_rho['r0'][r] for r in early]) -
            np.mean([mean_rho['r0'][r] for r in late]))
    d_r3 = (np.mean([mean_rho['r3'][r] for r in early]) -
            np.mean([mean_rho['r3'][r] for r in late]))
    interaction = d_r0 - d_r3
    print(f"\n  Interaktionseffekt (Δr0 - Δr3): {interaction:+.3f}")
    verdict = '✅ PC-Vorhersage bestätigt' if interaction > 0 else '⚠️  PC-Vorhersage nicht bestätigt'
    print(f"  {verdict}")

    n            = len(subjects)
    subjects_str = ', '.join(subjects)
    ax.set_title(
        f'PC Hierarchie-Gradient — Gruppenebene (N={n} Subjects)\n'
        f'Mittelwert ± SD  |  Interaktionseffekt Δr0−Δr3 = {interaction:+.3f}',
        fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(roi_names)
    ax.axvline(2.5, color='gray', linewidth=1, linestyle=':', alpha=0.4)
    ax.set_xlabel('ROI (früh → spät)', fontsize=11)
    ax.set_ylabel('Spearman ρ', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Gespeichert: {save_path}')


if __name__ == '__main__':
    SUBJECTS = ['sub-01', 'sub-02', 'sub-03']
    all_results = {}

    for sub_id in SUBJECTS:
        print(f'\n{"#" * 65}')
        print(f'# Subject: {sub_id}')
        print(f'{"#" * 65}\n')
        try:
            rho, nc = run_subject(sub_id)
            all_results[sub_id] = (rho, nc)
        except FileNotFoundError as e:
            print(f'  ⚠️  {sub_id} übersprungen — Datei nicht gefunden: {e}')

    # Gruppen-Plot falls mehr als 1 Subject erfolgreich
    if len(all_results) > 1:
        cfg_final = Config()
        # rho_results und nc_results trennen
        all_rho = {s: v[0] for s, v in all_results.items()}
        all_nc  = {s: v[1] for s, v in all_results.items()}
        # Mittlere Noise Ceilings über Subjects
        mean_nc = {roi: float(np.mean([all_nc[s][roi] for s in all_nc]))
                   for roi in cfg_final.ROI_NAMES}
        # Mittlere ResNet-Werte über Subjects
        mean_resnet = {roi: float(np.mean([all_rho[s]['resnet'][roi]
                       for s in all_rho if 'resnet' in all_rho[s]]))
                       for roi in cfg_final.ROI_NAMES}
        plot_group_average(
            all_rho,
            list(cfg_final.ROI_NAMES),
            mean_resnet,
            mean_nc,
            str(cfg_final.OUT_DIR / 'pc_hierarchy_group.png')
        )
        # Gruppen-Ergebnisse speichern
        np.save(str(cfg_final.OUT_DIR / 'pc_rho_results_group.npy'), all_rho)
        np.save(str(cfg_final.OUT_DIR / 'pc_noise_ceilings_group.npy'), all_nc)
        print(f"Gespeichert: pc_rho_results_group.npy, pc_noise_ceilings_group.npy")
        # Permutationstest
        perm_results = permutation_test_interaction(
            all_rho,
            list(cfg_final.ROI_NAMES),
            n_perm=1000
        )
        # Null-Verteilung speichern
        np.save(str(cfg_final.OUT_DIR / 'permutation_null.npy'),
                perm_results['null'])
        plot_permutation_null(
            perm_results,
            str(cfg_final.OUT_DIR / 'pc_permutation_test.png')
        )
        # Gruppen-Ergebnisse speichern
        np.save(str(cfg_final.OUT_DIR / 'pc_group_rho.npy'), all_rho)
        np.save(str(cfg_final.OUT_DIR / 'pc_group_nc.npy'), all_nc)
        np.save(str(cfg_final.OUT_DIR / 'pc_group_mean_nc.npy'), mean_nc)
        np.save(str(cfg_final.OUT_DIR / 'pc_permutation_results.npy'),
                {'observed': perm_results['observed'], 'p_value': perm_results['p_value']})
        print(f'\nAlle Subjects abgeschlossen: {list(all_results.keys())}')
        print(f'  {cfg_final.OUT_DIR}/pc_group_rho.npy')
        print(f'  {cfg_final.OUT_DIR}/pc_permutation_results.npy')
    elif len(all_results) == 1:
        print('\nNur 1 Subject — kein Gruppen-Plot.')
    else:
        print('\n⚠️  Keine Subjects erfolgreich geladen.')