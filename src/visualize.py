#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 15:55:56 2026

@author: danielgp
"""

# visualize.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generar_reporte_completo(
    histories_ablacion,
    results_ablacion,
    results_clahe_drive,
    results_baseline_chase,
    results_clahe_chase,
    model_baseline,
    test_loader,
    device,
    figuras_path='./figuras'
):
    """
    Genera y guarda todas las figuras y tablas del informe científico.
    """
    print("\n" + "="*80)
    print("GENERANDO FIGURAS PARA EL INFORME (CALIDAD PUBLICACIÓN)")
    print("="*80)
    
    os.makedirs(figuras_path, exist_ok=True)
    loss_names_display = {
        'BCE sola': 'BCE sola',
        'Dice sola': 'Dice sola',
        'BCE_Dice_ponderada': 'BCE + Dice (ponderada)'
    }

    # ----------------------------------------------------------------------------
    # FIGURA 1: Curvas de pérdida y Dice del estudio de ablación
    # ----------------------------------------------------------------------------
    if histories_ablacion:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = {'BCE sola': 'blue', 'Dice sola': 'green', 'BCE_Dice_ponderada': 'red'}

        for loss_key, history in histories_ablacion.items():
            label_name = loss_names_display[loss_key]
            axes[0].plot(history['val_loss'], label=label_name, color=colors[loss_key], linewidth=2)
            axes[1].plot(history['val_dice'], label=label_name, color=colors[loss_key], linewidth=2)

        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Pérdida de validación')
        axes[0].set_title('Evolución de la Pérdida')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Dice Score de validación')
        axes[1].set_title('Evolución del Dice Score')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(figuras_path, 'figura1_curvas_ablacion.png'), dpi=150)
        plt.close()
        print("✅ Figura 1 guardada: figura1_curvas_ablacion.png")

    # ----------------------------------------------------------------------------
    # FIGURA 2: Gráfico de barras del estudio de ablación
    # ----------------------------------------------------------------------------
    if results_ablacion:
        fig, ax = plt.subplots(figsize=(10, 6))
        loss_names = list(loss_names_display.values())
        loss_keys = list(loss_names_display.keys())
        x = np.arange(len(loss_names))
        width = 0.2

        metrics = ['Dice', 'Sensibilidad', 'Especificidad', 'AUC']
        colors_bar = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

        for i, metric in enumerate(metrics):
            values = [results_ablacion[key][metric.lower()] for key in loss_keys]
            ax.bar(x + i*width, values, width, label=metric, color=colors_bar[i])

        ax.set_xlabel('Función de Pérdida')
        ax.set_ylabel('Valor')
        ax.set_title('Comparación de Métricas por Función de Pérdida')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(loss_names)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(figuras_path, 'figura2_ablacion_barras.png'), dpi=150)
        plt.close()
        print("✅ Figura 2 guardada: figura2_ablacion_barras.png")

    # ----------------------------------------------------------------------------
    # FIGURA 3: Ejemplo de segmentación en DRIVE
    # ----------------------------------------------------------------------------
    model_baseline.eval()
    test_iter = iter(test_loader)
    img_sample, mask_sample = next(test_iter)
    img_sample = img_sample.to(device)

    with torch.no_grad():
        output = model_baseline(img_sample)
        probs = torch.sigmoid(output).cpu().numpy().squeeze()
        pred = (probs > 0.3).astype(np.uint8)

    img_np = img_sample.cpu().numpy().squeeze().transpose(1, 2, 0)
    mask_np = mask_sample.cpu().numpy().squeeze()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ['Imagen original', 'Máscara real', 'Mapa de prob.', 'Predicción (thr=0.3)']
    images = [img_np, mask_np, probs, pred]
    cmaps = [None, 'gray', 'hot', 'gray']

    for ax, title, img, cmap in zip(axes, titles, images, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(figuras_path, 'figura3_segmentacion_drive.png'), dpi=150)
    plt.close()
    print("✅ Figura 3 guardada: figura3_segmentacion_drive.png")

    # ----------------------------------------------------------------------------
    # FIGURA 4: Generalización DRIVE -> CHASE_DB1
    # ----------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics_names = ['Dice', 'Sensibilidad', 'Especificidad', 'AUC']
    
    drive_values = [
        results_ablacion['BCE_Dice_ponderada']['dice'],
        results_ablacion['BCE_Dice_ponderada']['sensibilidad'],
        results_ablacion['BCE_Dice_ponderada']['especificidad'],
        results_ablacion['BCE_Dice_ponderada']['auc']
    ]
    chase_values = [
        results_baseline_chase['dice'], results_baseline_chase['sensibilidad'],
        results_baseline_chase['especificidad'], results_baseline_chase['auc']
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    ax.bar(x - width/2, drive_values, width, label='DRIVE (test)', color='#2ecc71')
    ax.bar(x + width/2, chase_values, width, label='CHASE_DB1 (sin adaptación)', color='#e74c3c')

    ax.set_xlabel('Métrica')
    ax.set_ylabel('Valor')
    ax.set_title('Generalización: DRIVE → CHASE_DB1')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figuras_path, 'figura4_generalizacion.png'), dpi=150)
    plt.close()
    print("✅ Figura 4 guardada: figura4_generalizacion.png")

    # ----------------------------------------------------------------------------
    # FIGURA 5: Efecto de CLAHE en la generalización
    # ----------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    modelos = ['Baseline\nDRIVE', 'Baseline\nCHASE', 'CLAHE\nDRIVE', 'CLAHE\nCHASE']
    valores_dice = [
        results_ablacion['BCE_Dice_ponderada']['dice'],
        results_baseline_chase['dice'],
        results_clahe_drive['dice'],
        results_clahe_chase['dice']
    ]

    colors_dice = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
    bars = ax.bar(modelos, valores_dice, color=colors_dice, edgecolor='black', linewidth=1.5)

    for bar, valor in zip(bars, valores_dice):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{valor:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylim(0, 1)
    ax.set_ylabel('Dice Score')
    ax.set_title('Efecto de CLAHE en la Generalización')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(figuras_path, 'figura5_efecto_clahe.png'), dpi=150)
    plt.close()
    print("✅ Figura 5 guardada: figura5_efecto_clahe.png")

    # ----------------------------------------------------------------------------
    # FIGURA 6: Análisis cualitativo de fallos
    # ----------------------------------------------------------------------------
    best_sens, worst_sens = -1, 2
    best_data, worst_data = None, None

    for img, mask in test_loader:
        img_gpu = img.to(device)
        with torch.no_grad():
            output = model_baseline(img_gpu)
            probs = torch.sigmoid(output).cpu().numpy().squeeze()
            pred = (probs > 0.3).astype(np.uint8)
            mask_np = mask.numpy().squeeze()

            tp = np.sum((pred == 1) & (mask_np == 1))
            fn = np.sum((pred == 0) & (mask_np == 1))
            sens = tp / (tp + fn + 1e-8)

            if sens > best_sens:
                best_sens = sens
                best_data = (img.numpy().squeeze().transpose(1, 2, 0), mask_np, probs, pred)
            if sens < worst_sens:
                worst_sens = sens
                worst_data = (img.numpy().squeeze().transpose(1, 2, 0), mask_np, probs, pred)

    if best_data and worst_data:
        best_img, best_mask, best_probs, best_pred = best_data
        worst_img, worst_mask, worst_probs, worst_pred = worst_data

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # Buena segmentación
        axes[0,0].imshow(best_img); axes[0,0].set_title(f'Alta sens: {best_sens:.3f} (Gruesos)')
        axes[0,1].imshow(best_mask, cmap='gray'); axes[0,1].set_title('Máscara real')
        axes[0,2].imshow(best_probs, cmap='hot'); axes[0,2].set_title('Mapa de probabilidad')
        axes[0,3].imshow(best_pred, cmap='gray'); axes[0,3].set_title('Predicción')

        # Mala segmentación
        axes[1,0].imshow(worst_img); axes[1,0].set_title(f'Baja sens: {worst_sens:.3f} (Finos)')
        axes[1,1].imshow(worst_mask, cmap='gray')
        axes[1,2].imshow(worst_probs, cmap='hot')
        axes[1,3].imshow(worst_pred, cmap='gray')

        for ax in axes.flatten():
            ax.axis('off')

        plt.suptitle('Análisis de Fallos: Vasos Gruesos vs. Capilares Finos', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(figuras_path, 'figura6_analisis_fallos.png'), dpi=150)
        plt.close()
        print("✅ Figura 6 guardada: figura6_analisis_fallos.png")

    # ----------------------------------------------------------------------------
    # TABLA RESUMEN (.csv)
    # ----------------------------------------------------------------------------
    summary_data = {
        'Modelo/Pérdida': ['BCE sola', 'Dice sola', 'BCE + Dice (ponderada)', 'CLAHE (DRIVE)', 'Baseline (CHASE)', 'CLAHE (CHASE)'],
        'Dice': [
            results_ablacion['BCE sola']['dice'], results_ablacion['Dice sola']['dice'],
            results_ablacion['BCE_Dice_ponderada']['dice'], results_clahe_drive['dice'],
            results_baseline_chase['dice'], results_clahe_chase['dice']
        ],
        'Sensibilidad': [
            results_ablacion['BCE sola']['sensibilidad'], results_ablacion['Dice sola']['sensibilidad'],
            results_ablacion['BCE_Dice_ponderada']['sensibilidad'], results_clahe_drive['sensibilidad'],
            results_baseline_chase['sensibilidad'], results_clahe_chase['sensibilidad']
        ],
        'Especificidad': [
            results_ablacion['BCE sola']['especificidad'], results_ablacion['Dice sola']['especificidad'],
            results_ablacion['BCE_Dice_ponderada']['especificidad'], results_clahe_drive['especificidad'],
            results_baseline_chase['especificidad'], results_clahe_chase['especificidad']
        ],
        'AUC': [
            results_ablacion['BCE sola']['auc'], results_ablacion['Dice sola']['auc'],
            results_ablacion['BCE_Dice_ponderada']['auc'], results_clahe_drive['auc'],
            results_baseline_chase['auc'], results_clahe_chase['auc']
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    csv_path = os.path.join(figuras_path, 'tabla_resultados.csv')
    df_summary.to_csv(csv_path, index=False)
    print(f"✅ Tabla de métricas guardada en: {csv_path}")
    print("\n¡Listo! Imágenes listas para ser integradas a Overleaf.")