import os
os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 9, 'figure.dpi': 200, 'axes.titleweight': 'bold'})

print("Loading data...")
dt_raw = pd.read_excel('Resumen Downtime Current.xlsx', sheet_name='2024')

# Prepare downtime
dt = dt_raw.copy()
dt['Downtime_h'] = pd.to_numeric(dt['Downtime'], errors='coerce')
dt = dt.dropna(subset=['Downtime_h'])
dt['Mes'] = dt['Opened'].dt.to_period('M')

# 1. Figura 2: Distribución del Downtime (horas y log10)
fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
d_clip = dt['Downtime_h'].clip(upper=6)
ax[0].hist(d_clip, bins=50, color='#4C72B0', edgecolor='white', alpha=0.9)
ax[0].set_title('(a) Horas (recortado a 6 h)', fontsize=9, fontweight='bold')
ax[0].set_xlabel('Horas', fontsize=8)
ax[0].set_ylabel('Paros', fontsize=8)
ax[0].grid(axis='y', linestyle=':', alpha=0.6)

log_d = np.log10(dt.loc[dt['Downtime_h'] > 0, 'Downtime_h'])
ax[1].hist(log_d, bins=50, color='#DD8452', edgecolor='white', alpha=0.9)
ax[1].set_title('(b) Escala logarítmica', fontsize=9, fontweight='bold')
ax[1].set_xlabel(r'$\log_{10}(\text{horas})$', fontsize=8)
ax[1].set_ylabel('Paros', fontsize=8)
ax[1].grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('fig_downtime_dist.png', dpi=300)
plt.close()
print("Saved fig_downtime_dist.png")

# 2. Figura 3: Dispersión MTTR vs Paros (Confiabilidad)
dias_periodo = (dt.Opened.max() - dt.Opened.min()).days
rel = dt.groupby('Equipo').agg(fallas=('Downtime_h','count'), horas_paro=('Downtime_h','sum'), MTTR_h=('Downtime_h','mean'))
rel['MTBF_h'] = (dias_periodo*24 - rel['horas_paro']) / rel['fallas']

fig, ax = plt.subplots(figsize=(5.5, 3.6))
# normalize sizes
s_norm = (rel['horas_paro'] / rel['horas_paro'].max()) * 400 + 25
ax.scatter(rel['fallas'], rel['MTTR_h'], s=s_norm, alpha=0.6, color='#4C72B0', edgecolors='none')
for n in ['Las Vegas', 'Charros', 'LaGloriosa', 'Consentida', 'LaBarranca']:
    if n in rel.index:
        r = rel.loc[n]
        ax.annotate(n, (r.fallas + 6, r.MTTR_h - 0.02), fontsize=7.5, fontweight='bold')
ax.axvline(rel.fallas.median(), ls='--', c='gray', alpha=0.7)
ax.axhline(rel.MTTR_h.median(), ls='--', c='gray', alpha=0.7)
ax.set_title('Número de paros vs. MTTR por máquina', fontsize=9, fontweight='bold')
ax.set_xlabel('Número de paros', fontsize=8)
ax.set_ylabel('MTTR (h)', fontsize=8)
ax.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig('fig_mttr_scatter.png', dpi=300)
plt.close()
print("Saved fig_mttr_scatter.png")

# 3. Figura 4: Evolución temporal mensual
m = dt.groupby('Mes')['Downtime_h'].agg(eventos='count', horas='sum')
m = m.iloc[:-1] if dt.Opened.max().day < 28 else m

fig, ax = plt.subplots(figsize=(6.5, 3.0))
meses_str = [str(x)[2:] for x in m.index]
ax.bar(meses_str, m['horas'], color='#4C72B0', label='Horas de paro', alpha=0.85, width=0.6)
ax.tick_params(axis='x', rotation=45, labelsize=7.5)
ax.set_ylabel('Horas de paro', fontsize=8, color='#4C72B0')
ax.grid(axis='y', linestyle=':', alpha=0.5)
ax2 = ax.twinx()
ax2.plot(meses_str, m['eventos'], color='#C44E52', marker='o', linewidth=2, label='Número de paros')
ax2.set_ylabel('Número de paros', fontsize=8, color='#C44E52')
ax2.grid(False)
ax.set_title('Horas de paro (barras) y número de paros (línea) por mes', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_series_tiempo.png', dpi=300)
plt.close()
print("Saved fig_series_tiempo.png")

print("All plots generated!")
