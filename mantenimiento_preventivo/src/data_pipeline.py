"""
data_pipeline.py
Pipeline de procesamiento de datos, ingeniería de variables y partición temporal
para el proyecto de Mantenimiento Predictivo (Semana 1).
"""

import os
import time
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict


def load_raw_data(data_dir: str = 'mantenimiento_preventivo/data/raw') -> Dict[str, pd.DataFrame]:
    """
    Carga los 5 archivos CSV originales del dataset Microsoft Azure Predictive Maintenance.
    """
    print(f"Cargando archivos CSV originales desde {data_dir}...")
    telemetry = pd.read_csv(os.path.join(data_dir, 'PdM_telemetry.csv'))
    errors = pd.read_csv(os.path.join(data_dir, 'PdM_errors.csv'))
    maint = pd.read_csv(os.path.join(data_dir, 'PdM_maint.csv'))
    failures = pd.read_csv(os.path.join(data_dir, 'PdM_failures.csv'))
    machines = pd.read_csv(os.path.join(data_dir, 'PdM_machines.csv'))

    # Convertir columnas de tiempo
    telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])
    errors['datetime'] = pd.to_datetime(errors['datetime'])
    maint['datetime'] = pd.to_datetime(maint['datetime'])
    failures['datetime'] = pd.to_datetime(failures['datetime'])

    print(f"✓ Telemetría: {len(telemetry):,} registros | Máquinas: {len(machines)} | "
          f"Errores: {len(errors):,} | Mantenimientos: {len(maint):,} | Fallas: {len(failures):,}")
    return {
        'telemetry': telemetry,
        'errors': errors,
        'maint': maint,
        'failures': failures,
        'machines': machines
    }


def compute_telemetry_features(telemetry: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula promedios y desviaciones estándar móviles para ventanas de 3h y 24h.
    Garantiza causalidad estricta (ventanas cerradas a la derecha [t-W+1, t] sin lookahead).
    """
    print("Calculando estadísticas móviles de telemetría (3h y 24h)...")
    df = telemetry.sort_values(['machineID', 'datetime']).reset_index(drop=True)
    sensors = ['volt', 'rotate', 'pressure', 'vibration']

    for s in sensors:
        grp = df.groupby('machineID')[s]
        df[f'{s}_mean_3h'] = grp.transform(lambda x: x.rolling(3, min_periods=1).mean())
        df[f'{s}_std_3h'] = grp.transform(lambda x: x.rolling(3, min_periods=1).std()).fillna(0.0)
        df[f'{s}_mean_24h'] = grp.transform(lambda x: x.rolling(24, min_periods=1).mean())
        df[f'{s}_std_24h'] = grp.transform(lambda x: x.rolling(24, min_periods=1).std()).fillna(0.0)

    return df


def compute_error_features(telemetry: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el conteo de errores por tipo (error1 a error5) en la ventana móvil causal de las últimas 24h.
    """
    print("Calculando conteo de errores en las últimas 24h...")
    # Pivotear errores por fecha y máquina
    error_pivot = errors.pivot_table(
        index=['datetime', 'machineID'],
        columns='errorID',
        aggfunc='size',
        fill_value=0
    ).reset_index()

    df = pd.merge(telemetry, error_pivot, on=['datetime', 'machineID'], how='left').fillna(0)

    error_cols = [f'error{i}' for i in range(1, 6)]
    for col in error_cols:
        if col not in df.columns:
            df[col] = 0
        df[f'{col}_count_24h'] = df.groupby('machineID')[col].transform(
            lambda x: x.rolling(24, min_periods=1).sum()
        )

    # Conteo acumulado total de errores en 24h
    df['total_errors_24h'] = df[[f'{col}_count_24h' for col in error_cols]].sum(axis=1)
    return df


def compute_maintenance_features(telemetry: pd.DataFrame, maint: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula las horas transcurridas desde el último reemplazo para cada componente (comp1 a comp4).
    Utiliza pd.merge_asof backward para emparejar estrictamente con registros de mantenimiento <= t.
    Esto modela el desgaste acumulado y previene fugas de información.
    """
    print("Calculando horas transcurridas desde último reemplazo de componente...")
    df = telemetry.sort_values('datetime').reset_index(drop=True)
    comps = ['comp1', 'comp2', 'comp3', 'comp4']

    for comp in comps:
        m_c = maint[maint['comp'] == comp].sort_values('datetime')[['datetime', 'machineID']].copy()
        m_c[f'maint_dt_{comp}'] = m_c['datetime']
        df = pd.merge_asof(
            df,
            m_c[['datetime', 'machineID', f'maint_dt_{comp}']],
            on='datetime',
            by='machineID',
            direction='backward'
        )
        df[f'hours_since_{comp}'] = (df['datetime'] - df[f'maint_dt_{comp}']).dt.total_seconds() / 3600.0
        df = df.drop(columns=[f'maint_dt_{comp}'])

    return df


def add_static_features(telemetry: pd.DataFrame, machines: pd.DataFrame) -> pd.DataFrame:
    """
    Incorpora metadatos estáticos de la máquina: modelo (one-hot) y antigüedad (años).
    """
    print("Incorporando metadatos estáticos de máquina (modelo y antigüedad)...")
    df = pd.merge(telemetry, machines, on='machineID', how='left')
    model_dummies = pd.get_dummies(df['model'], prefix='model', drop_first=False, dtype=int)
    df = pd.concat([df, model_dummies], axis=1)
    return df


def construct_labels(telemetry: pd.DataFrame, failures: pd.DataFrame, horizon_hours: int = 24) -> pd.DataFrame:
    """
    Construye las etiquetas con horizonte temporal H (por defecto H=24 horas):
    - Tarea principal (binaria): 1 si ocurre alguna falla en (t, t + H], 0 si no.
    - Tarea secundaria (multiclase): 0: Ninguno, 1: comp1, 2: comp2, 3: comp3, 4: comp4.
    """
    print(f"Construyendo etiquetas con horizonte H={horizon_hours}h...")
    df = telemetry.sort_values(['machineID', 'datetime']).reset_index(drop=True)

    # Pivot de fallas por fecha y máquina
    fail_pivot = failures.pivot_table(
        index=['datetime', 'machineID'],
        columns='failure',
        aggfunc='size',
        fill_value=0
    ).reset_index()

    df = pd.merge(df, fail_pivot, on=['datetime', 'machineID'], how='left').fillna(0)

    comps = ['comp1', 'comp2', 'comp3', 'comp4']
    for c in comps:
        if c not in df.columns:
            df[c] = 0
        # Rolling forward causal en ventana inversa: suma de fallas en (t, t + H]
        df[f'{c}_next24'] = df.groupby('machineID')[c].transform(
            lambda s: s.iloc[::-1].rolling(horizon_hours, min_periods=1).sum().iloc[::-1].shift(-1).fillna(0)
        )

    # Etiqueta binaria
    df['label_bin'] = (df[[f'{c}_next24' for c in comps]].sum(axis=1) > 0).astype(int)

    # Etiqueta multiclase: 0: Ninguno, 1: comp1, 2: comp2, 3: comp3, 4: comp4
    df['label_multi'] = 0
    for idx, c in enumerate(comps, 1):
        mask = (df[f'{c}_next24'] > 0) & (df['label_multi'] == 0)
        df.loc[mask, 'label_multi'] = idx

    return df


def create_temporal_splits(df: pd.DataFrame, margin_hours: int = 24) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Realiza la partición temporal fija obligatoria:
    - Entrenamiento: 1 ene 2015 – 31 ago 2015 (con margen de H=24h -> t hasta 2015-08-30 23:00)
    - Validación: 1 sep 2015 – 31 oct 2015 (con margen de H=24h -> t hasta 2015-10-30 23:00)
    - Prueba: 1 nov 2015 – 31 dic 2015 (con margen de H=24h -> t hasta 2015-12-30 23:00)
    """
    print(f"Generando partición temporal fija con margen de seguridad H={margin_hours}h...")
    train_end = pd.to_datetime('2015-08-31 23:00:00') - pd.Timedelta(hours=margin_hours)
    val_start = pd.to_datetime('2015-09-01 00:00:00')
    val_end = pd.to_datetime('2015-10-31 23:00:00') - pd.Timedelta(hours=margin_hours)
    test_start = pd.to_datetime('2015-11-01 00:00:00')
    test_end = pd.to_datetime('2015-12-31 23:00:00') - pd.Timedelta(hours=margin_hours)

    train_df = df[(df['datetime'] >= '2015-01-01') & (df['datetime'] <= train_end)].copy()
    val_df = df[(df['datetime'] >= val_start) & (df['datetime'] <= val_end)].copy()
    test_df = df[(df['datetime'] >= test_start) & (df['datetime'] <= test_end)].copy()

    print(f"✓ Entrenamiento: {len(train_df):,} filas ({train_df['datetime'].min()} a {train_df['datetime'].max()})")
    print(f"✓ Validación:    {len(val_df):,} filas ({val_df['datetime'].min()} a {val_df['datetime'].max()})")
    print(f"✓ Prueba:        {len(test_df):,} filas ({test_df['datetime'].min()} a {test_df['datetime'].max()})")

    return train_df, val_df, test_df


def get_feature_columns() -> List[str]:
    """Retorna la lista de columnas de características para los modelos de línea base."""
    sensors = ['volt', 'rotate', 'pressure', 'vibration']
    comps = ['comp1', 'comp2', 'comp3', 'comp4']
    error_cols = [f'error{i}' for i in range(1, 6)]

    features = (
        [f'{s}_mean_3h' for s in sensors] +
        [f'{s}_std_3h' for s in sensors] +
        [f'{s}_mean_24h' for s in sensors] +
        [f'{s}_std_24h' for s in sensors] +
        [f'{col}_count_24h' for col in error_cols] +
        ['total_errors_24h'] +
        [f'hours_since_{c}' for c in comps] +
        ['age'] +
        [f'model_model{i}' for i in range(1, 5)]
    )
    return features


def generate_imbalance_table(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera la tabla comparativa de desbalance de clases entre Entrenamiento y Validación.
    """
    classes = [
        (0, 'Ninguna Falla (Sano)'),
        (1, 'comp1 (Eléctrico)'),
        (2, 'comp2 (Hidráulico)'),
        (3, 'comp3 (Mecánico)'),
        (4, 'comp4 (Vibración/Rodamiento)')
    ]

    records = []
    # Clases individuales
    for cls_id, label in classes:
        tr_cnt = (train_df['label_multi'] == cls_id).sum()
        tr_pct = tr_cnt / len(train_df) * 100.0
        val_cnt = (val_df['label_multi'] == cls_id).sum()
        val_pct = val_cnt / len(val_df) * 100.0
        records.append({
            'Etiqueta': label,
            'Clase': cls_id,
            'Train (N)': tr_cnt,
            'Train (%)': f"{tr_pct:.2f}%",
            'Val (N)': val_cnt,
            'Val (%)': f"{val_pct:.2f}%"
        })

    # Total Positivos Binarios (Falla Global)
    tr_bin_cnt = (train_df['label_bin'] == 1).sum()
    tr_bin_pct = tr_bin_cnt / len(train_df) * 100.0
    val_bin_cnt = (val_df['label_bin'] == 1).sum()
    val_bin_pct = val_bin_cnt / len(val_df) * 100.0

    records.append({
        'Etiqueta': 'TOTAL POSITIVOS (Binario: Falla en H=24h)',
        'Clase': '1-4',
        'Train (N)': tr_bin_cnt,
        'Train (%)': f"{tr_bin_pct:.2f}%",
        'Val (N)': val_bin_cnt,
        'Val (%)': f"{val_bin_pct:.2f}%"
    })

    return pd.DataFrame(records)


def run_pipeline(
    raw_dir: str = 'mantenimiento_preventivo/data/raw',
    processed_dir: str = 'mantenimiento_preventivo/data/processed'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Ejecuta el pipeline completo y exporta los conjuntos procesados.
    """
    t0 = time.time()
    os.makedirs(processed_dir, exist_ok=True)

    data = load_raw_data(raw_dir)
    df = compute_telemetry_features(data['telemetry'])
    df = compute_error_features(df, data['errors'])
    df = compute_maintenance_features(df, data['maint'])
    df = add_static_features(df, data['machines'])
    df = construct_labels(df, data['failures'], horizon_hours=24)

    train_df, val_df, test_df = create_temporal_splits(df, margin_hours=24)
    feature_cols = get_feature_columns()

    # Guardar en processed en formato comprimido parquet o csv
    train_path = os.path.join(processed_dir, 'train.csv')
    val_path = os.path.join(processed_dir, 'val.csv')
    test_path = os.path.join(processed_dir, 'test.csv')

    print(f"Exportando conjuntos a {processed_dir}...")
    # Exportamos las columnas relevantes para optimizar espacio
    cols_to_save = ['datetime', 'machineID', 'label_bin', 'label_multi'] + feature_cols
    train_df[cols_to_save].to_csv(train_path, index=False)
    val_df[cols_to_save].to_csv(val_path, index=False)
    test_df[cols_to_save].to_csv(test_path, index=False)

    # Imbalance table
    imbalance_df = generate_imbalance_table(train_df, val_df)
    imbalance_df.to_csv(os.path.join(processed_dir, 'imbalance_table.csv'), index=False)
    print("\n--- TABLA DE DESBALANCE DE CLASES ---")
    print(imbalance_df.to_string(index=False))

    print(f"\n✓ Pipeline completado exitosamente en {time.time()-t0:.2f} segundos.")
    return train_df, val_df, test_df, feature_cols


if __name__ == '__main__':
    run_pipeline()
