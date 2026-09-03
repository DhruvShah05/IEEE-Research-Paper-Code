"""
train/train_neural.py — Generic PyTorch training loop for DeepLOB, Transformer,
and StructuredTransformer.

Changes (3.1):
  - Early stopping with configurable patience on validation Macro-F1
    (max_epochs: 50, patience: 5 as defaults).
  - AdamW optimizer with weight_decay (configurable).
  - Learning-rate scheduler: cosine or ReduceLROnPlateau (configurable).
  - Gradient clipping (configurable).
  - FocalLoss: class weights passed as alpha (was None).
  - 'no mitigation' imbalance strategy path.
  - Deterministic DataLoader (generator and worker_init_fn) — now in main.py.
  - Saves test_probs.npy (shape N×3) via the returned model (main.py calls inference).
  - Saves model.pt (best checkpoint) and param_count → run_manifest.json.
  - Records wall-clock training time and per-batch inference latency.
"""

import copy
import json
import logging
import os
import time

import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval.metrics import compute_all_metrics

# ---------------------------------------------------------------------------
# GPU detection helpers
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    """Returns the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _gpu_info() -> str:
    """Returns a human-readable GPU info string."""
    if not torch.cuda.is_available():
        return 'No CUDA GPU'
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    vram  = [torch.cuda.get_device_properties(i).total_memory / 1024**3 for i in range(n)]
    return '  '.join(f'{name} ({vram_gb:.0f} GB)'
                     for name, vram_gb in zip(names, vram))

logger = logging.getLogger(__name__)

MODEL_MODULE_MAP = {
    'deeplob': 'deeplob',
    'transformer': 'transformer',
    'structured_transformer': 'structured_transformer',
}


class FocalLoss(nn.Module):
    """
    Standard focal loss (Lin et al., 2017) for addressing class imbalance.
    gamma=0 recovers standard cross-entropy.

    Fix 3.1: ``alpha`` (per-class weights) is now properly wired — when
    ``strategy='focal_loss'`` the class weights are computed from training
    labels and passed as ``alpha`` so the loss is both focal *and* weighted.
    """
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # 1D tensor of per-class weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def _build_criterion(strategy: str, imbalance_cfg: dict, train_loader: DataLoader,
                     device: torch.device) -> nn.Module:
    """Constructs the loss criterion based on the imbalance strategy."""
    if strategy == 'class_weight':
        all_y = torch.cat([y for _, y in train_loader]).numpy()
        class_counts = np.bincount(all_y, minlength=3)
        weights = 1.0 / np.maximum(class_counts, 1)
        weights = weights / weights.sum() * len(class_counts)
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        logger.info(f"WeightedCE — class weights: {weights.tolist()}")
        return nn.CrossEntropyLoss(weight=class_weights_tensor)

    elif strategy == 'focal_loss':
        gamma = imbalance_cfg.get('focal_gamma', 2.0)
        # Fix 3.1: compute and pass alpha (class weights) to focal loss
        all_y = torch.cat([y for _, y in train_loader]).numpy()
        class_counts = np.bincount(all_y, minlength=3)
        weights = 1.0 / np.maximum(class_counts, 1)
        weights = weights / weights.sum() * len(class_counts)
        alpha_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        logger.info(f"FocalLoss(gamma={gamma}) with alpha={weights.tolist()}")
        return FocalLoss(gamma=gamma, alpha=alpha_tensor)

    else:  # 'none' — no imbalance mitigation (required for ablation §7.2)
        logger.info("CrossEntropyLoss (no imbalance mitigation).")
        return nn.CrossEntropyLoss()


def train_neural_model(
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    run_dir: str,
) -> nn.Module:
    """
    Generic PyTorch training loop — optimised for RTX 5080 / high-VRAM GPUs.

    GPU optimisations (auto-detected):
      - torch.compile()   : kernel fusion (PyTorch ≥ 2.0; ~10–30% speedup on Transformer)
      - AMP (bf16)        : automatic mixed precision with bfloat16 on Ampere+/Blackwell;
                            halves VRAM and speeds up matmul by ~2×
      - DataParallel      : uses all available GPUs automatically
      - pin_memory        : already set in DataLoader calls in main.py
      - TF32              : enabled on Ampere+ (default in PyTorch ≥ 1.11)
      - cudnn.benchmark   : enabled when input sizes are fixed (Transformer/DeepLOB)

    Saves per run:
      - training_history.json  : per-epoch loss + val Macro-F1
      - config_used.json       : full resolved config
      - model.pt               : best checkpoint by val Macro-F1

    Returns the model loaded with the best-epoch weights.
    """
    device = _get_device()
    gpu_info = _gpu_info()
    logger.info(f"Training on device: {device}  ({gpu_info})")

    # --- TF32 (Ampere/Blackwell: RTX 30xx/40xx/50xx) ---
    # bfloat16 TF32 enables faster matrix multiplication with minimal precision loss.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        # benchmark=True: cudnn selects the fastest algorithm for fixed input shapes.
        # Only safe when the input shape is constant across batches.
        torch.backends.cudnn.benchmark = True
        logger.info("TF32 and cudnn.benchmark enabled")

    # Save full config for traceability
    with open(os.path.join(run_dir, 'config_used.json'), 'w') as f:
        json.dump(config, f, indent=4)

    model_name  = config['model']
    module_name = MODEL_MODULE_MAP.get(model_name, model_name)
    model_module = importlib.import_module(f"models.{module_name}")
    model = model_module.build_model(config).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model {model_name} — parameter count: {param_count:,}")

    # --- Multi-GPU (DataParallel) ---
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # --- torch.compile (PyTorch ≥ 2.0 — RTX 5080 benefits significantly) ---
    gpu_cfg = config.get('gpu', {})
    use_compile = gpu_cfg.get('compile', True)   # on by default for CUDA
    if use_compile and torch.cuda.is_available():
        try:
            compile_mode = gpu_cfg.get('compile_mode', 'reduce-overhead')
            model = torch.compile(model, mode=compile_mode)
            logger.info(f"torch.compile enabled (mode='{compile_mode}')")
        except Exception as e:
            logger.warning(f"torch.compile failed ({e}) — falling back to eager mode")

    # --- AMP scaler (bfloat16 on Blackwell/Ampere, float16 fallback) ---
    use_amp = gpu_cfg.get('amp', True) and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # B3: torch.cuda.amp.GradScaler is deprecated in PyTorch >= 2.5; use torch.amp.GradScaler.
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))
    if use_amp:
        logger.info(f"AMP enabled (dtype={amp_dtype})")

    train_cfg = config.get('training', {})
    max_epochs = train_cfg.get('max_epochs', train_cfg.get('epochs', 50))
    patience   = train_cfg.get('patience', 5)
    lr         = train_cfg.get('learning_rate', 1e-3)
    weight_decay = train_cfg.get('weight_decay', 1e-4)
    grad_clip  = train_cfg.get('grad_clip', 1.0)   # 0 or None = disabled
    scheduler_type = train_cfg.get('scheduler', 'cosine')  # 'cosine' | 'plateau' | 'none'

    # --- Optimizer (AdamW with weight_decay — fix 3.1) ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- Scheduler (fix 3.1) ---
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=lr * 0.01
        )
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=False
        )
    else:
        scheduler = None

    # --- Loss criterion ---
    imbalance_cfg = config.get('imbalance', {})
    strategy = imbalance_cfg.get('strategy', 'none')
    criterion = _build_criterion(strategy, imbalance_cfg, train_loader, device)

    best_val_f1 = -1.0
    best_model_state = None
    epochs_no_improve = 0
    history = []

    train_start = time.time()
    epoch_bar = tqdm(range(max_epochs), desc=f"Training {model_name}", unit="epoch")

    for epoch in epoch_bar:
        # --- Training phase ---
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            # non_blocking transfer — overlaps data copy with GPU compute
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()

            # AMP forward pass
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=use_amp):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

            # AMP backward pass
            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            train_loss += loss.item()

        # --- Validation phase ---
        model.eval()
        val_preds, val_true = [], []
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=use_amp):
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())

        metrics = compute_all_metrics(np.array(val_true), np.array(val_preds))
        macro_f1 = metrics['macro_f1']

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)

        epoch_bar.set_postfix({
            'train_loss': f"{avg_train_loss:.4f}",
            'val_loss':   f"{avg_val_loss:.4f}",
            'val_f1':     f"{macro_f1:.4f}",
        })

        # Scheduler step (fix 3.1)
        if scheduler is not None:
            if scheduler_type == 'plateau':
                scheduler.step(macro_f1)
            else:
                scheduler.step()

        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss':   avg_val_loss,
            'val_macro_f1': macro_f1,
            'lr': optimizer.param_groups[0]['lr'],
        })

        # Best-checkpoint selection by val Macro-F1
        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping (fix 3.1)
        if patience and epochs_no_improve >= patience:
            logger.info(
                f"Early stopping at epoch {epoch + 1} "
                f"(no improvement for {patience} epochs). Best val F1={best_val_f1:.4f}"
            )
            break

    wall_clock_s = time.time() - train_start
    logger.info(
        f"Training complete in {wall_clock_s:.1f}s. Best val Macro-F1: {best_val_f1:.4f}"
    )

    # Restore best weights
    model.load_state_dict(best_model_state)

    # Save per-epoch training history
    with open(os.path.join(run_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    # Save model checkpoint (fix 3.1)
    model_path = os.path.join(run_dir, 'model.pt')
    torch.save(best_model_state, model_path)
    logger.info(f"Best model checkpoint saved to {model_path}")

    # Update run_manifest with param_count and wall-clock time (fix 3.1)
    manifest_path = os.path.join(run_dir, 'run_manifest.json')
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    else:
        manifest = {}
    manifest['param_count'] = param_count
    manifest['training_wall_clock_s'] = wall_clock_s
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)

    return model
