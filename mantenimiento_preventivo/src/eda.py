"""
eda.py
Módulo de Análisis Exploratorio de Datos (EDA) y generación de figuras de alta calidad
para el reporte académico del proyecto de Mantenimiento Predictivo (Semana 1).
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_academic_style():
    """Configura el estilo visual académico profesional para las figuras."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'legend.fontsize': 9.5,
        'figure.titlesize': 13,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def generate_figure1_sensor_distributions(raw_dir: str, output_path: str):
    """
    Figura 1: Distribución de cada sensor (voltaje, rotación, presión, vibración)
    por modelo de máquina (model1 a model4).
    """
    print("Generando Figura 1: Distribución de sensores por modelo de máquina...")
    telemetry = pd.read_csv(os.path.join(raw_dir, 'PdM_telemetry.csv'), nrows=200000)
    machines = pd.read_csv(os.path.join(raw_dir, 'PdM_machines.csv'))
    df = pd.merge(telemetry, machines, on='machineID', how='left')

    sensors = [
        ('volt', 'Voltaje (V)', '#1f77b4'),
        ('rotate', 'Rotación (RPM)', '#ff7f0e'),
        ('pressure', 'Presión (psi)', '#2ca02c'),
        ('vibration', 'Vibración (mm/s)', '#d62728')
    ]
    models = ['model1', 'model2', 'model3', 'model4']

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    axes = axes.flatten()

    for idx, (sensor, label, color) in enumerate(sensors):
        ax = axes[idx]
        data_to_plot = [df[df['model'] == m][sensor].dropna().values for m in models]
        bp = ax.boxplot(
            data_to_plot,
            tick_labels=['Mod 1', 'Mod 2', 'Mod 3', 'Mod 4'],
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color='black', linewidth=1.5),
            boxprops=dict(facecolor=color, alpha=0.6, edgecolor=color),
            whiskerprops=dict(color='#555555', linewidth=1.2),
            capprops=dict(color='#555555', linewidth=1.2)
        )
        ax.set_title(f'Distribución de {label}', fontweight='bold')
        ax.set_ylabel(label)
        ax.set_xlabel('Modelo de Máquina')
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('Distribución Operativa de Sensores por Modelo de Máquina', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Guardada: {output_path}")


def generate_figure2_event_frequencies(raw_dir: str, output_path: str):
    """
    Figura 2: Frecuencia de errores no fatales, mantenimientos y fallas reales
    desglosadas por componente y tipo.
    """
    print("Generando Figura 2: Frecuencia de eventos operativos y desbalance...")
    errors = pd.read_csv(os.path.join(raw_dir, 'PdM_errors.csv'))
    maint = pd.read_csv(os.path.join(raw_dir, 'PdM_maint.csv'))
    failures = pd.read_csv(os.path.join(raw_dir, 'PdM_failures.csv'))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # 1. Errores
    err_counts = errors['errorID'].value_counts().sort_index()
    axes[0].bar(err_counts.index, err_counts.values, color='#4575b4', edgecolor='black', alpha=0.85)
    axes[0].set_title('Errores No Fatales por Tipo', fontweight='bold')
    axes[0].set_ylabel('Frecuencia Anual')
    axes[0].set_xlabel('Código de Error')
    for i, v in enumerate(err_counts.values):
        axes[0].text(i, v + 25, f"{v:,}", ha='center', fontsize=9, fontweight='bold')
    axes[0].set_ylim(0, max(err_counts.values) * 1.15)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # 2. Mantenimientos
    maint_counts = maint['comp'].value_counts().sort_index()
    axes[1].bar(maint_counts.index, maint_counts.values, color='#74add1', edgecolor='black', alpha=0.85)
    axes[1].set_title('Mantenimientos (Reemplazos)', fontweight='bold')
    axes[1].set_ylabel('Frecuencia Anual')
    axes[1].set_xlabel('Componente')
    for i, v in enumerate(maint_counts.values):
        axes[1].text(i, v + 25, f"{v:,}", ha='center', fontsize=9, fontweight='bold')
    axes[1].set_ylim(0, max(maint_counts.values) * 1.15)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # 3. Fallas Reales Catastróficas
    fail_counts = failures['failure'].value_counts().sort_index()
    axes[2].bar(fail_counts.index, fail_counts.values, color='#d73027', edgecolor='black', alpha=0.85)
    axes[2].set_title('Fallas Reales por Componente', fontweight='bold')
    axes[2].set_ylabel('Frecuencia Anual')
    axes[2].set_xlabel('Componente Crítico')
    for i, v in enumerate(fail_counts.values):
        axes[2].text(i, v + 8, f"{v:,}", ha='center', fontsize=9, fontweight='bold')
    axes[2].set_ylim(0, max(fail_counts.values) * 1.18)
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('Comparación de Frecuencias: Errores, Mantenimientos y Fallas Catastróficas (2015)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Guardada: {output_path}")


def generate_figure3_pre_failure_telemetry(raw_dir: str, output_path: str):
    """
    Figura 3: Comportamiento dinámico de la telemetría en las 72 horas previas a una falla
    frente a periodos sanos de operación, evidenciando las firmas físicas de degradación.
    """
    print("Generando Figura 3: Telemetría 72h pre-falla vs periodos sanos...")
    telemetry = pd.read_csv(os.path.join(raw_dir, 'PdM_telemetry.csv'))
    failures = pd.read_csv(os.path.join(raw_dir, 'PdM_failures.csv'))
    telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])
    failures['datetime'] = pd.to_datetime(failures['datetime'])

    # Extraer ventanas de 72h antes de cada falla
    pre_fail_windows = []
    for _, row in failures.iterrows():
        m_id = row['machineID']
        f_dt = row['datetime']
        comp = row['failure']
        start_dt = f_dt - pd.Timedelta(hours=72)
        win = telemetry[(telemetry['machineID'] == m_id) & (telemetry['datetime'] >= start_dt) & (telemetry['datetime'] <= f_dt)].copy()
        if len(win) >= 48:
            win['hours_to_failure'] = (win['datetime'] - f_dt).dt.total_seconds() / 3600.0
            win['failed_comp'] = comp
            pre_fail_windows.append(win)

    df_pre = pd.concat(pre_fail_windows, ignore_index=True)

    # Medias de periodos sanos basales (overall means)
    healthy_means = telemetry[['volt', 'rotate', 'pressure', 'vibration']].mean()

    # Mapeo de componente a sensor más afectado físicamente:
    # comp1 -> volt, comp2 -> pressure, comp3 -> rotate, comp4 -> vibration
    sensor_map = [
        ('comp1', 'volt', 'Voltaje (V)', 'Falla Comp 1 (Eléctrico)', '#1f77b4'),
        ('comp2', 'pressure', 'Presión (psi)', 'Falla Comp 2 (Hidráulico)', '#2ca02c'),
        ('comp3', 'rotate', 'Rotación (RPM)', 'Falla Comp 3 (Mecánico)', '#ff7f0e'),
        ('comp4', 'vibration', 'Vibración (mm/s)', 'Falla Comp 4 (Rodamientos)', '#d62728')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for idx, (comp, sensor, label, title, color) in enumerate(sensor_map):
        ax = axes[idx]
        sub = df_pre[df_pre['failed_comp'] == comp]
        agg = sub.groupby('hours_to_failure')[sensor].agg(['mean', 'std']).reset_index()

        # Línea temporal previa
        ax.plot(agg['hours_to_failure'], agg['mean'], color=color, linewidth=2.2, label='Pre-falla (Promedio)')
        ax.fill_between(
            agg['hours_to_failure'],
            agg['mean'] - 0.75 * agg['std'],
            agg['mean'] + 0.75 * agg['std'],
            color=color,
            alpha=0.25,
            label='Dispersión (±0.75σ)'
        )

        # Línea de referencia sana
        h_val = healthy_means[sensor]
        ax.axhline(h_val, color='black', linestyle='--', linewidth=1.5, label=f'Media Operación Sana ({h_val:.1f})')
        ax.axvline(0, color='red', linestyle=':', linewidth=1.8, label='Momento de Falla (t=0)')

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Horas restantes hasta la falla (t=0)')
        ax.set_ylabel(label)
        ax.set_xlim(-72, 0)
        ax.legend(loc='upper left', frameon=True, fontsize=8.5)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('Evolución de Sensores en las 72 Horas Previas a Fallas Críticas vs. Operación Sana',
                 fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Guardada: {output_path}")


def run_eda(
    raw_dir: str = 'mantenimiento_preventivo/data/raw',
    figures_dir: str = 'mantenimiento_preventivo/figures'
):
    """Ejecuta el análisis exploratorio y exporta las tres figuras clave."""
    set_academic_style()
    os.makedirs(figures_dir, exist_ok=True)

    fig1_path = os.path.join(figures_dir, 'fig1_distribucion_sensores_modelo.png')
    fig2_path = os.path.join(figures_dir, 'fig2_frecuencia_eventos_desbalance.png')
    fig3_path = os.path.join(figures_dir, 'fig3_telemetria_pre_falla_vs_sano.png')

    generate_figure1_sensor_distributions(raw_dir, fig1_path)
    generate_figure2_event_frequencies(raw_dir, fig2_path)
    generate_figure3_pre_failure_telemetry(raw_dir, fig3_path)
    print("\n✓ Todas las figuras exploratorias han sido generadas exitosamente.")


if __name__ == '__main__':
    run_eda()
