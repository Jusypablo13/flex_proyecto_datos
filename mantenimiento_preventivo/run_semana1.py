#!/usr/bin/env python3
"""
run_semana1.py
Script maestro de ejecución integral reproducible para la Entrega 1:
1. Pipeline de procesamiento de datos y partición temporal fija (Train, Val, Test).
2. Generación de figuras del análisis exploratorio (Figuras 1, 2 y 3).
3. Entrenamiento y evaluación de líneas base (Regresión Logística y GBDT), métricas y Figuras 4 y 5.
4. Generación del reporte académico en PDF (Reporte_Entrega1_PdM.pdf).
"""

import sys
import os
import time

# Asegurar que el directorio de src esté en el sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'mantenimiento_preventivo', 'src')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'mantenimiento_preventivo', 'reports')
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, REPORTS_DIR)

from data_pipeline import run_pipeline
from eda import run_eda
from baseline_models import train_and_evaluate_baselines
from generar_reporte_pdf import build_pdf_report


def main():
    t_total = time.time()
    print("=" * 80)
    print("PROYECTO DE MANTENIMIENTO PREDICTIVO INDUSTRIAL — ENTREGA 1 (SEMANA 1)")
    print("Tecnológico de Monterrey | Machine Learning & Deep Learning Industrial")
    print("=" * 80)

    # 1. Pipeline de datos
    print("\n[PASO 1/4] Ejecutando Pipeline de Datos y Partición Temporal...")
    train_df, val_df, test_df, feature_cols = run_pipeline(
        raw_dir=os.path.join(PROJECT_ROOT, 'mantenimiento_preventivo', 'data', 'raw'),
        processed_dir=os.path.join(PROJECT_ROOT, 'mantenimiento_preventivo', 'data', 'processed')
    )

    # 2. Análisis exploratorio y figuras
    print("\n[PASO 2/4] Generando Figuras de Análisis Exploratorio (EDA)...")
    run_eda(
        raw_dir=os.path.join(PROJECT_ROOT, 'mantenimiento_preventivo', 'data', 'raw'),
        figures_dir=os.path.join(PROJECT_ROOT, 'mantenimiento_preventivo', 'figures')
    )

    # 3. Modelado y evaluación de línea base
    print("\n[PASO 3/4] Entrenando Modelos de Línea Base y Evaluando en Validación...")
    metrics = train_and_evaluate_baselines(
        processed_dir=os.path.join(PROJECT_ROOT, 'mantenimiento_preventivo', 'data', 'processed'),
        figures_dir=os.path.join(PROJECT_ROOT, 'mantenimiento_preventivo', 'figures'),
        reports_dir=os.path.join(PROJECT_ROOT, 'mantenimiento_preventivo', 'reports')
    )

    # 4. Generación del reporte académico oficial en PDF
    print("\n[PASO 4/4] Compilando Reporte Académico Oficial en PDF (Máx. 5 páginas)...")
    build_pdf_report(
        reports_dir=os.path.join(PROJECT_ROOT, 'mantenimiento_preventivo', 'reports'),
        figures_dir=os.path.join(PROJECT_ROOT, 'mantenimiento_preventivo', 'figures'),
        output_filename='Reporte_Entrega1_PdM.pdf'
    )

    print("\n" + "=" * 80)
    print(f"✓ PROCESO COMPLETADO EXITOSAMENTE EN {time.time()-t_total:.2f} SEGUNDOS")
    print(f"  • Reporte PDF:    {os.path.join(REPORTS_DIR, 'Reporte_Entrega1_PdM.pdf')}")
    print(f"  • Reporte LaTeX:  {os.path.join(REPORTS_DIR, 'Reporte_Entrega1_PdM.tex')}")
    print(f"  • Figuras (300DPI): {os.path.join(PROJECT_ROOT, 'mantenimiento_preventivo', 'figures')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
