#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 15:49:02 2026

@author: danielgp
"""

# utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

# ----- Funciones de Pérdida -----
class BCELoss(nn.Module):
    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        return F.binary_cross_entropy(inputs, targets, reduction='mean')

class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        return 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

class DiceBCELossConPesos(nn.Module):
    def __init__(self, peso_vasos=30.0):
        super(DiceBCELossConPesos, self).__init__()
        self.peso_vasos = peso_vasos

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        peso = torch.where(targets == 1,
                          torch.tensor(self.peso_vasos, device=targets.device),
                          torch.tensor(1.0, device=targets.device))
        bce_loss = F.binary_cross_entropy(inputs, targets, weight=peso, reduction='mean')
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return bce_loss + dice_loss

# ----- Funciones de Entrenamiento y Evaluación -----
def entrenar_modelo(modelo, criterion, train_loader, val_loader, device,
                    epochs=200, lr=3e-5, patience=50, name="modelo", verbose=True):
    optimizer = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_dice = 0.0
    best_model_state = None
    patience_counter = 0
    best_epoch = 0

    history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': [], 'lr': []}

    for epoch in range(epochs):
        modelo.train()
        train_loss, train_dice = 0.0, 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = modelo(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                preds_flat, targets_flat = preds.view(-1), targets.view(-1)
                tp = ((preds_flat == 1) & (targets_flat == 1)).sum().float()
                fp = ((preds_flat == 1) & (targets_flat == 0)).sum().float()
                fn = ((preds_flat == 0) & (targets_flat == 1)).sum().float()
                dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
                train_dice += dice.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)

        # Validación
        modelo.eval()
        val_loss, val_dice = 0.0, 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = modelo(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                preds_flat, targets_flat = preds.view(-1), targets.view(-1)
                tp = ((preds_flat == 1) & (targets_flat == 1)).sum().float()
                fp = ((preds_flat == 1) & (targets_flat == 0)).sum().float()
                fn = ((preds_flat == 0) & (targets_flat == 1)).sum().float()
                dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
                val_dice += dice.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_dice'].append(avg_train_dice)
        history['val_dice'].append(avg_val_dice)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        scheduler.step(avg_val_loss)

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_model_state = {k: v.cpu().clone() for k, v in modelo.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch + 1
            if verbose: print(f"  ✅ Mejora! Nuevo best Dice: {best_val_dice:.4f} (época {epoch+1})")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose: print(f"  🛑 Early stopping activado (época {epoch+1})")
            break

    if best_model_state is not None:
        modelo.load_state_dict(best_model_state)

    return history, best_val_dice, best_model_state

def evaluar_modelo(modelo, dataloader, device, umbral=0.3):
    modelo.eval()
    all_probs, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = modelo(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_targets.extend(targets.cpu().numpy().flatten())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    preds = (all_probs > umbral).astype(int)

    tp = np.sum((preds == 1) & (all_targets == 1))
    tn = np.sum((preds == 0) & (all_targets == 0))
    fp = np.sum((preds == 1) & (all_targets == 0))
    fn = np.sum((preds == 0) & (all_targets == 1))

    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    sensibilidad = tp / (tp + fn + 1e-8)
    especificidad = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    auc = roc_auc_score(all_targets, all_probs)

    return {'dice': dice, 'sensibilidad': sensibilidad, 'especificidad': especificidad,
            'precision': precision, 'auc': auc, 'probs': all_probs, 'targets': all_targets}