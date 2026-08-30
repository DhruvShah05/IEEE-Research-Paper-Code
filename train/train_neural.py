import os
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import importlib
import copy
from tqdm import tqdm

from eval.metrics import compute_all_metrics

logger = logging.getLogger(__name__)

# Maps config 'model' name to actual module filename (mirrors train_tree.py convention)
MODEL_MODULE_MAP = {
    'deeplob': 'deeplob',
    'transformer': 'transformer',
    'structured_transformer': 'structured_transformer',
}

class FocalLoss(nn.Module):
    """
    Standard focal loss (Lin et al., 2017) for addressing class imbalance.
    gamma=0 recovers standard cross-entropy. Configurable via config imbalance.focal_gamma.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
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


def train_neural_model(config: dict, train_loader: DataLoader, val_loader: DataLoader, run_dir: str):
    """
    Generic PyTorch training loop shared by DeepLOB, Transformer, and StructuredTransformer.

    Saves per run (build.md §6):
      - training_history.json  : per-epoch loss + val Macro-F1
      - config_used.json       : full resolved config (hyperparams, seed, market, model)
    Best checkpoint selected by validation Macro-F1.
    """
    # Device priority: CUDA (cloud GPU) > MPS (Apple Silicon M-series) > CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.info(f"Training on device: {device}")

    # Save the full resolved config for traceability (build.md §6)
    with open(os.path.join(run_dir, 'config_used.json'), 'w') as f:
        json.dump(config, f, indent=4)

    model_name = config['model']
    module_name = MODEL_MODULE_MAP.get(model_name, model_name)
    model_module = importlib.import_module(f"models.{module_name}")
    model = model_module.build_model(config).to(device)

    train_config = config.get('training', {})
    epochs = train_config.get('epochs', 15)
    lr = train_config.get('learning_rate', 0.001)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Imbalance strategy (build.md §7.2) ---
    imbalance_cfg = config.get('imbalance', {})
    strategy = imbalance_cfg.get('strategy', 'none')

    criterion = nn.CrossEntropyLoss()
    if strategy == 'class_weight':
        # Compute class weights from training labels — fit on training data only (build.md §8 rule 7)
        all_y = torch.cat([y for _, y in train_loader]).numpy()
        class_counts = np.bincount(all_y, minlength=3)
        # Inverse-frequency weighting, normalised so average weight = 1
        weights = 1.0 / np.maximum(class_counts, 1)
        weights = weights / weights.sum() * len(class_counts)
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        logger.info(f"Using Weighted CrossEntropy. Class weights: {weights.tolist()}")
    elif strategy == 'focal_loss':
        gamma = imbalance_cfg.get('focal_gamma', 2.0)
        criterion = FocalLoss(gamma=gamma)
        logger.info(f"Using Focal Loss with gamma={gamma}")

    best_val_f1 = -1.0
    best_model_state = None
    history = []

    epoch_bar = tqdm(range(epochs), desc=f"Training {model_name}", unit="epoch")
    for epoch in epoch_bar:
        # --- Training phase ---
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # --- Validation phase ---
        model.eval()
        val_preds, val_true = [], []
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())

        metrics = compute_all_metrics(np.array(val_true), np.array(val_preds))
        macro_f1 = metrics['macro_f1']

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        epoch_bar.set_postfix({
            'train_loss': f"{avg_train_loss:.4f}",
            'val_loss': f"{avg_val_loss:.4f}",
            'val_macro_f1': f"{macro_f1:.4f}"
        })

        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_macro_f1': macro_f1
        })

        # Best-checkpoint selection by val Macro-F1 (build.md §6)
        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            best_model_state = copy.deepcopy(model.state_dict())

    logger.info(f"Training complete. Best val Macro-F1: {best_val_f1:.4f}")

    # Restore best weights
    model.load_state_dict(best_model_state)

    # Save per-epoch training history (build.md §6)
    with open(os.path.join(run_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    return model

