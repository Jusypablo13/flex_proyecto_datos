"""
baseline_models.py
Entrenamiento y evaluación rigurosa de modelos de línea base clásica (Semana 1):
- Regresión Logística
- Gradient Boosting (HistGradientBoosting / LightGBM)
Cálculo de métricas en validación:
- PR-AUC
- ROC-AUC
- Recall con Precisión >= 0.80
- Matriz de confusión multiclase (5x5)
Generación de Figuras 4 y 5.
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report
)

from data_pipeline import get_feature_columns


def train_and_evaluate_baselines(
    processed_dir: str = 'mantenimiento_preventivo/data/processed',
    figures_dir: str = 'mantenimiento_preventivo/figures',
    reports_dir: str = 'mantenimiento_preventivo/reports'
):
    print("=" * 70)
    print("ENTRENAMIENTO Y EVALUACIÓN DE LÍNEA BASE EN CONJUNTO DE VALIDACIÓN")
    print("=" * 70)

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Cargar datos procesados
    print(f"Cargando conjuntos de datos desde {processed_dir}...")
    train_path = os.path.join(processed_dir, 'train.csv')
    val_path = os.path.join(processed_dir, 'val.csv')

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    feature_cols = [c for c in get_feature_columns() if c in train_df.columns]
    print(f"Características utilizadas ({len(feature_cols)}): {feature_cols}")

    X_train = train_df[feature_cols].values
    y_train_bin = train_df['label_bin'].values
    y_train_multi = train_df['label_multi'].values

    X_val = val_df[feature_cols].values
    y_val_bin = val_df['label_bin'].values
    y_val_multi = val_df['label_multi'].values

    print(f"Train: {X_train.shape[0]:,} muestras (Positivos: {y_train_bin.sum():,}, {y_train_bin.mean()*100:.2f}%)")
    print(f"Val:   {X_val.shape[0]:,} muestras (Positivos: {y_val_bin.sum():,}, {y_val_bin.mean()*100:.2f}%)")

    # -------------------------------------------------------------
    # 2. Escalamiento sin fuga (Fit solo en Train)
    # -------------------------------------------------------------
    print("\nAjustando StandardScaler exclusivamente en conjunto de entrenamiento...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # -------------------------------------------------------------
    # 3. Modelo 1: Regresión Logística (Binario)
    # -------------------------------------------------------------
    print("\n--- Entrenando Modelo 1: Regresión Logística (Binario) ---")
    t0 = time.time()
    lr = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
    lr.fit(X_train_scaled, y_train_bin)
    t_lr_train = time.time() - t0

    lr_probs = lr.predict_proba(X_val_scaled)[:, 1]
    pr_auc_lr = float(average_precision_score(y_val_bin, lr_probs))
    roc_auc_lr = float(roc_auc_score(y_val_bin, lr_probs))

    p_lr, r_lr, th_lr = precision_recall_curve(y_val_bin, lr_probs)
    # Recall a Precision >= 0.80
    valid_p80_lr = (p_lr >= 0.80)
    if valid_p80_lr.any():
        idx_p80_lr = np.where(valid_p80_lr)[0]
        # max recall
        rec_p80_lr = float(r_lr[valid_p80_lr].max())
        best_p80_idx = idx_p80_lr[np.argmax(r_lr[valid_p80_lr])]
        th_p80_lr = float(th_lr[best_p80_idx]) if best_p80_idx < len(th_lr) else float(th_lr[-1])
        prec_at_rec_lr = float(p_lr[best_p80_idx])
    else:
        rec_p80_lr = 0.0
        th_p80_lr = 0.5
        prec_at_rec_lr = float(p_lr.max())

    print(f"✓ Regresión Logística entrenada en {t_lr_train:.2f}s")
    print(f"  PR-AUC: {pr_auc_lr:.4f} | ROC-AUC: {roc_auc_lr:.4f} | Recall (Prec>=0.80): {rec_p80_lr:.4f} (Umbral: {th_p80_lr:.4f})")

    # -------------------------------------------------------------
    # 4. Modelo 2: Gradient Boosting Decision Trees (Binario)
    # -------------------------------------------------------------
    print("\n--- Entrenando Modelo 2: Gradient Boosting (Binario) ---")
    t0 = time.time()
    # HistGradientBoostingClassifier es el análogo scikit-learn nativo ultrarrápido de LightGBM
    gbdt = HistGradientBoostingClassifier(
        class_weight='balanced',
        max_iter=150,
        learning_rate=0.1,
        random_state=42
    )
    gbdt.fit(X_train, y_train_bin)
    t_gbdt_train = time.time() - t0

    gbdt_probs = gbdt.predict_proba(X_val)[:, 1]
    pr_auc_gbdt = float(average_precision_score(y_val_bin, gbdt_probs))
    roc_auc_gbdt = float(roc_auc_score(y_val_bin, gbdt_probs))

    p_gbdt, r_gbdt, th_gbdt = precision_recall_curve(y_val_bin, gbdt_probs)
    valid_p80_gbdt = (p_gbdt >= 0.80)
    if valid_p80_gbdt.any():
        idx_p80_gbdt = np.where(valid_p80_gbdt)[0]
        rec_p80_gbdt = float(r_gbdt[valid_p80_gbdt].max())
        best_p80_idx_g = idx_p80_gbdt[np.argmax(r_gbdt[valid_p80_gbdt])]
        th_p80_gbdt = float(th_gbdt[best_p80_idx_g]) if best_p80_idx_g < len(th_gbdt) else float(th_gbdt[-1])
        prec_at_rec_gbdt = float(p_gbdt[best_p80_idx_g])
    else:
        rec_p80_gbdt = 0.0
        th_p80_gbdt = 0.5
        prec_at_rec_gbdt = float(p_gbdt.max())

    print(f"✓ Gradient Boosting entrenado en {t_gbdt_train:.2f}s")
    print(f"  PR-AUC: {pr_auc_gbdt:.4f} | ROC-AUC: {roc_auc_gbdt:.4f} | Recall (Prec>=0.80): {rec_p80_gbdt:.4f} (Umbral: {th_p80_gbdt:.4f})")

    # -------------------------------------------------------------
    # 5. Modelo Multiclase (5 clases: 0: Ninguno, 1..4: comp1..comp4)
    # -------------------------------------------------------------
    print("\n--- Entrenando Modelo Multiclase (Gradient Boosting 5 clases) ---")
    t0 = time.time()
    gbdt_multi = HistGradientBoostingClassifier(
        class_weight='balanced',
        max_iter=100,
        learning_rate=0.1,
        random_state=42
    )
    gbdt_multi.fit(X_train, y_train_multi)
    t_multi_train = time.time() - t0

    preds_multi = gbdt_multi.predict(X_val)
    cm_multi = confusion_matrix(y_val_multi, preds_multi)
    cm_multi_norm = confusion_matrix(y_val_multi, preds_multi, normalize='true')

    class_names = ['Ninguno', 'Comp 1', 'Comp 2', 'Comp 3', 'Comp 4']
    rep_dict = classification_report(y_val_multi, preds_multi, target_names=class_names, output_dict=True)

    print(f"✓ GBDT Multiclase entrenado en {t_multi_train:.2f}s")
    print("\nMatriz de Confusión Absoluta:")
    print(cm_multi)
    print("\nReporte de Clasificación Multiclase en Validación:")
    print(classification_report(y_val_multi, preds_multi, target_names=class_names, digits=4))

    # También evaluamos Regresión Logística Multiclase para comparación
    print("\n--- Entrenando Regresión Logística Multiclase ---")
    lr_multi = LogisticRegression(class_weight='balanced', max_iter=300, random_state=42)
    lr_multi.fit(X_train_scaled, y_train_multi)
    preds_lr_multi = lr_multi.predict(X_val_scaled)
    cm_lr_multi = confusion_matrix(y_val_multi, preds_lr_multi)
    rep_lr_dict = classification_report(y_val_multi, preds_lr_multi, target_names=class_names, output_dict=True)

    # -------------------------------------------------------------
    # 6. Generación de Figura 4: Curvas PR y ROC en Validación
    # -------------------------------------------------------------
    print("\nGenerando Figura 4: Curvas PR y ROC de validación...")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))

    # Precision-Recall Curve
    axes[0].plot(r_gbdt, p_gbdt, color='#2b83ba', linewidth=2.2, label=f'Gradient Boosting (PR-AUC = {pr_auc_gbdt:.4f})')
    axes[0].plot(r_lr, p_lr, color='#fdae61', linewidth=2.0, linestyle='--', label=f'Regresión Logística (PR-AUC = {pr_auc_lr:.4f})')
    axes[0].axhline(0.80, color='red', linestyle=':', linewidth=1.5, label='Requisito Precisión ≥ 0.80')

    # Marcador de punto de operación
    if valid_p80_gbdt.any():
        axes[0].scatter([rec_p80_gbdt], [prec_at_rec_gbdt], color='black', s=60, zorder=5,
                        label=f'Pto GBDT: Rec={rec_p80_gbdt:.3f}, Prec={prec_at_rec_gbdt:.3f}')
    if valid_p80_lr.any():
        axes[0].scatter([rec_p80_lr], [prec_at_rec_lr], color='#d7191c', s=60, zorder=5,
                        label=f'Pto LR: Rec={rec_p80_lr:.3f}, Prec={prec_at_rec_lr:.3f}')

    axes[0].set_title('Curva Precision-Recall (Validación H=24h)', fontweight='bold')
    axes[0].set_xlabel('Recall (Exhaustividad)')
    axes[0].set_ylabel('Precision (Precisión)')
    axes[0].set_xlim(0, 1.02)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc='lower left', frameon=True, fontsize=8.5)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # ROC Curve
    fpr_gbdt, tpr_gbdt, _ = roc_curve(y_val_bin, gbdt_probs)
    fpr_lr, tpr_lr, _ = roc_curve(y_val_bin, lr_probs)

    axes[1].plot(fpr_gbdt, tpr_gbdt, color='#2b83ba', linewidth=2.2, label=f'Gradient Boosting (ROC-AUC = {roc_auc_gbdt:.4f})')
    axes[1].plot(fpr_lr, tpr_lr, color='#fdae61', linewidth=2.0, linestyle='--', label=f'Regresión Logística (ROC-AUC = {roc_auc_lr:.4f})')
    axes[1].plot([0, 1], [0, 1], color='gray', linestyle=':', label='Azar (AUC = 0.5000)')

    axes[1].set_title('Curva ROC (Validación H=24h)', fontweight='bold')
    axes[1].set_xlabel('Tasa de Falsos Positivos (FPR)')
    axes[1].set_ylabel('Tasa de Verdaderos Positivos (TPR / Recall)')
    axes[1].set_xlim(0, 1.02)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc='lower right', frameon=True, fontsize=8.5)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('Rendimiento en Validación: Curvas Precision-Recall y ROC (Línea Base)',
                 fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, 'fig4_curvas_pr_roc_validacion.png')
    plt.savefig(fig4_path)
    plt.close()
    print(f"✓ Guardada: {fig4_path}")

    # -------------------------------------------------------------
    # 7. Generación de Figura 5: Matriz de Confusión Multiclase
    # -------------------------------------------------------------
    print("\nGenerando Figura 5: Matriz de Confusión Multiclase...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    # Matriz Absoluta
    im0 = axes[0].imshow(cm_multi, cmap='Blues', aspect='auto')
    axes[0].set_title('Matriz de Confusión Absoluta (GBDT)', fontweight='bold')
    axes[0].set_xlabel('Clase Predicha')
    axes[0].set_ylabel('Clase Real')
    axes[0].set_xticks(range(len(class_names)))
    axes[0].set_yticks(range(len(class_names)))
    axes[0].set_xticklabels(class_names, rotation=30, ha='right')
    axes[0].set_yticklabels(class_names)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_multi[i, j]
            color = 'white' if val > cm_multi.max() / 2 else 'black'
            axes[0].text(j, i, f"{val:,}", ha='center', va='center', color=color, fontweight='bold', fontsize=9)

    # Matriz Normalizada por Fila (Recall por clase)
    im1 = axes[1].imshow(cm_multi_norm, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1.0)
    axes[1].set_title('Matriz de Confusión Normalizada (Recall %)', fontweight='bold')
    axes[1].set_xlabel('Clase Predicha')
    axes[1].set_ylabel('Clase Real')
    axes[1].set_xticks(range(len(class_names)))
    axes[1].set_yticks(range(len(class_names)))
    axes[1].set_xticklabels(class_names, rotation=30, ha='right')
    axes[1].set_yticklabels(class_names)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_multi_norm[i, j]
            color = 'white' if val > 0.5 else 'black'
            axes[1].text(j, i, f"{val*100:.1f}%", ha='center', va='center', color=color, fontweight='bold', fontsize=9)

    plt.suptitle('Evaluación Multiclase en Validación: Identificación del Componente Crítico a Fallar en H=24h',
                 fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout()
    fig5_path = os.path.join(figures_dir, 'fig5_matriz_confusion_multiclase.png')
    plt.savefig(fig5_path)
    plt.close()
    print(f"✓ Guardada: {fig5_path}")

    # -------------------------------------------------------------
    # 8. Exportación de Métricas a JSON y CSV
    # -------------------------------------------------------------
    results = {
        'modelos_binarios': {
            'regresion_logistica': {
                'pr_auc': pr_auc_lr,
                'roc_auc': roc_auc_lr,
                'recall_at_precision_80': rec_p80_lr,
                'precision_at_recall': prec_at_rec_lr,
                'threshold_precision_80': th_p80_lr,
                'train_time_sec': t_lr_train
            },
            'gradient_boosting': {
                'pr_auc': pr_auc_gbdt,
                'roc_auc': roc_auc_gbdt,
                'recall_at_precision_80': rec_p80_gbdt,
                'precision_at_recall': prec_at_rec_gbdt,
                'threshold_precision_80': th_p80_gbdt,
                'train_time_sec': t_gbdt_train
            }
        },
        'multiclase_gbdt': {
            'accuracy': rep_dict['accuracy'],
            'macro_f1': rep_dict['macro avg']['f1-score'],
            'weighted_f1': rep_dict['weighted avg']['f1-score'],
            'por_clase': {
                cls_name: {
                    'precision': rep_dict[cls_name]['precision'],
                    'recall': rep_dict[cls_name]['recall'],
                    'f1_score': rep_dict[cls_name]['f1-score'],
                    'support': int(rep_dict[cls_name]['support'])
                }
                for cls_name in class_names
            },
            'confusion_matrix': cm_multi.tolist()
        },
        'multiclase_lr': {
            'accuracy': rep_lr_dict['accuracy'],
            'macro_f1': rep_lr_dict['macro avg']['f1-score'],
            'weighted_f1': rep_lr_dict['weighted avg']['f1-score'],
            'por_clase': {
                cls_name: {
                    'precision': rep_lr_dict[cls_name]['precision'],
                    'recall': rep_lr_dict[cls_name]['recall'],
                    'f1_score': rep_lr_dict[cls_name]['f1-score'],
                    'support': int(rep_lr_dict[cls_name]['support'])
                }
                for cls_name in class_names
            },
            'confusion_matrix': cm_lr_multi.tolist()
        }
    }

    metrics_json_path = os.path.join(reports_dir, 'metrics_summary.json')
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Métricas exportadas a {metrics_json_path}")

    # Tabla resumen en CSV
    summary_table = pd.DataFrame([
        {
            'Modelo': 'Regresión Logística (Lineal)',
            'PR-AUC': f"{pr_auc_lr:.4f}",
            'ROC-AUC': f"{roc_auc_lr:.4f}",
            'Recall (Prec≥0.80)': f"{rec_p80_lr:.4f}",
            'Umbral Óptimo': f"{th_p80_lr:.4f}",
            'Macro F1 (Multiclase)': f"{rep_lr_dict['macro avg']['f1-score']:.4f}",
            'Tiempo Entr. (s)': f"{t_lr_train:.2f}"
        },
        {
            'Modelo': 'Gradient Boosting (HistGBDT)',
            'PR-AUC': f"{pr_auc_gbdt:.4f}",
            'ROC-AUC': f"{roc_auc_gbdt:.4f}",
            'Recall (Prec≥0.80)': f"{rec_p80_gbdt:.4f}",
            'Umbral Óptimo': f"{th_p80_gbdt:.4f}",
            'Macro F1 (Multiclase)': f"{rep_dict['macro avg']['f1-score']:.4f}",
            'Tiempo Entr. (s)': f"{t_gbdt_train:.2f}"
        }
    ])
    metrics_csv_path = os.path.join(reports_dir, 'tabla_resultados_baseline.csv')
    summary_table.to_csv(metrics_csv_path, index=False)
    print(f"✓ Tabla resumen exportada a {metrics_csv_path}")
    print("\n--- RESUMEN DE MODELOS EN VALIDACIÓN ---")
    print(summary_table.to_string(index=False))

    return results


if __name__ == '__main__':
    train_and_evaluate_baselines()
