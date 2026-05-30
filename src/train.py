#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 15:49:28 2026

@author: danielgp
"""

# train.py
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Importaciones modulares locales
from model import UNet
from dataset import DRIVEDataset, ChaseDataset, AplicarCLAHE
from utils import BCELoss, DiceLoss, DiceBCELossConPesos, entrenar_modelo, evaluar_modelo
from visualize import generar_reporte_completo

def main():
    print("="*80)
    print("SEGMENTACIÓN DE VASOS RETINIANOS CON U-NET - ANÁLISIS COMPLETO")
    print("="*80)

    # Rutas locales (adaptadas para GitHub/Local)
    base_path = '.'
    os.makedirs(os.path.join(base_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'figuras'), exist_ok=True)
    ruta_drive = os.path.join(base_path, 'data', 'raw', 'DRIVE')
    ruta_chase = os.path.join(base_path, 'data', 'raw', 'CHASE_DB1')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")

    # 1. Transformaciones
    transform_baseline_img = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    transform_clahe_img = transforms.Compose([AplicarCLAHE(clip_limit=2.0), transforms.Resize((512, 512)), transforms.ToTensor()])
    transform_mask = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])

    # 2. Datasets y DataLoaders
    # NOTA: Asegúrate de tener tus datos en ./data/raw/DRIVE y ./data/raw/CHASE_DB1 antes de ejecutar esto
    try:
        full_drive_dataset = DRIVEDataset(ruta_drive, transform_img=transform_baseline_img, transform_mask=transform_mask)
        full_drive_dataset_clahe = DRIVEDataset(ruta_drive, transform_img=transform_clahe_img, transform_mask=transform_mask)
    except FileNotFoundError:
        print(f"ERROR: No se encontraron los datos en {ruta_drive}. Descárgalos primero.")
        return

    torch.manual_seed(42)
    total = len(full_drive_dataset)
    train_size = int(0.6 * total)
    val_size = int(0.2 * total)
    test_size = total - train_size - val_size

    train_subset_baseline, val_subset_baseline, test_dataset = torch.utils.data.random_split(
        full_drive_dataset, [train_size, val_size, test_size])
    train_subset_clahe, val_subset_clahe, _ = torch.utils.data.random_split(
        full_drive_dataset_clahe, [train_size, val_size, test_size])

    BATCH_SIZE = 1
    train_loader = DataLoader(train_subset_baseline, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset_baseline, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    train_loader_clahe = DataLoader(train_subset_clahe, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_clahe = DataLoader(val_subset_clahe, batch_size=BATCH_SIZE, shuffle=False)

    chase_loader = None
    if os.path.exists(ruta_chase):
        chase_dataset = ChaseDataset(ruta_chase, transform_img=transform_baseline_img, transform_mask=transform_mask)
        chase_loader = DataLoader(chase_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Estudio de Ablación
    loss_configs = {
        'BCE sola': BCELoss(),
        'Dice sola': DiceLoss(),
        'BCE_Dice_ponderada': DiceBCELossConPesos(peso_vasos=30.0)
    }

    results_ablacion = {}
    for loss_key, criterion in loss_configs.items():
        print(f"\n--- Entrenando con {loss_key} ---")
        model = UNet().to(device)
        history, best_dice, _ = entrenar_modelo(model, criterion, train_loader, val_loader, device, name=loss_key)
        torch.save(model.state_dict(), os.path.join(base_path, 'models', f'unet_{loss_key}.pth'))
        results_ablacion[loss_key] = evaluar_modelo(model, test_loader, device)

    # 4. Entrenamiento CLAHE
    print("\n--- Entrenando CLAHE ---")
    model_clahe = UNet().to(device)
    criterion_clahe = DiceBCELossConPesos(peso_vasos=30.0)
    history_clahe, best_dice_clahe, _ = entrenar_modelo(model_clahe, criterion_clahe, train_loader_clahe, val_loader_clahe, device, name="CLAHE")
    torch.save(model_clahe.state_dict(), os.path.join(base_path, 'models', 'unet_clahe.pth'))
    results_clahe_drive = evaluar_modelo(model_clahe, test_loader, device)

# 5. Evaluar CHASE para los diccionarios (Manejando fallback si no hay CHASE)
    results_baseline_chase = {'dice': 0.0, 'sensibilidad': 0.0, 'especificidad': 0.0, 'auc': 0.0}
    results_clahe_chase = {'dice': 0.0, 'sensibilidad': 0.0, 'especificidad': 0.0, 'auc': 0.0}
    if chase_loader is not None:
        results_baseline_chase = evaluar_modelo(model, chase_loader, device)
        results_clahe_chase = evaluar_modelo(model_clahe, chase_loader, device)
    else:
        print("⚠️ CHASE_DB1 no disponible - usando valores en 0.0 para las figuras")

    # 6. Generación de Figuras delegada a visualize.py
    histories_ablacion = {} # Si deseas guardar las curvas de loss/dice en el bucle del paso 3, guárdalas en este diccionario.
    
    generar_reporte_completo(
        histories_ablacion=histories_ablacion, 
        results_ablacion=results_ablacion,
        results_clahe_drive=results_clahe_drive,
        results_baseline_chase=results_baseline_chase,
        results_clahe_chase=results_clahe_chase,
        model_baseline=model, # El modelo entrenado con BCE_Dice_ponderada
        test_loader=test_loader,
        device=device,
        figuras_path=os.path.join(base_path, 'figuras')
    )

if __name__ == "__main__":
    main()
    
    