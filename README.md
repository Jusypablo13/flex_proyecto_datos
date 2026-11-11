# Optimización de Mantenimiento y Analítica Predictiva Industrial
**Tecnológico de Monterrey | Maestría / Ingeniería en Ciencia de Datos y Machine Learning Industrial**

---

## 👥 Equipo de Trabajo
* **Juan Pablo Torres Guillen** — Matrícula: *A01743268*
* **Angel David Lugo Lafarga** — Matrícula: *A01741564*
* **Jose Pablo Lopez Rodriguez** — Matrícula: *A01638911*

---

## 📁 Organización del Repositorio (Estructura Limpia)

Para evitar desorden y "código spaghetti", el repositorio se encuentra modularizado en dos proyectos independientes:

```text
flex_proyecto/
│
├── mantenimiento_preventivo/       # 🚀 NUEVO RETO: Predicción de Fallas en Máquinas (Semana 1)
│   ├── data/
│   │   ├── raw/                    # CSVs originales (telemetría, errores, mantenimientos, fallas, máquinas)
│   │   └── processed/              # Conjuntos procesados y tabla de desbalance (train, val, test)
│   ├── src/
│   │   ├── __init__.py
│   │   ├── data_pipeline.py        # Pipeline de ingeniería de variables y partición temporal fija
│   │   ├── eda.py                  # Generación de figuras exploratorias (Figuras 1, 2 y 3)
│   │   └── baseline_models.py      # Entrenamiento de Regresión Logística y GBDT (Figuras 4 y 5)
│   ├── figures/                    # Figuras de alta resolución (300 DPI) para publicación
│   │   ├── fig1_distribucion_sensores_modelo.png
│   │   ├── fig2_frecuencia_eventos_desbalance.png
│   │   ├── fig3_telemetria_pre_falla_vs_sano.png
│   │   ├── fig4_curvas_pr_roc_validacion.png
│   │   └── fig5_matriz_confusion_multiclase.png
│   ├── reports/
│   │   ├── Reporte_Entrega1_PdM.pdf # 📄 REPORTE OFICIAL EN PDF (Exactamente 5 páginas)
│   │   ├── Reporte_Entrega1_PdM.tex # Código fuente en formato IEEE Conference
│   │   ├── generar_reporte_pdf.py   # Script de compilación ReportLab
│   │   ├── metrics_summary.json     # Métricas completas exportadas
│   │   └── tabla_resultados_baseline.csv
│   ├── notebooks/
│   │   └── 01_analisis_exploratorio_y_baseline.ipynb # Notebook interactivo documentado
│   ├── requirements.txt
│   └── run_semana1.py              # Ejecución local en la carpeta
│
├── proyecto_tickets_flex/          # 📦 PROYECTO PREVIO: Clasificación de Tickets Flex
│   ├── Clasificacion de Tickets 2021.xlsm
│   ├── clasificador_tickets.py
│   ├── EDA_Flex.ipynb
│   ├── reporte_final_ieee_actualizado.tex
│   └── ...
│
├── run_semana1.py                  # ⚡ Script maestro ejecutable de un solo comando
├── requirements.txt                # Dependencias generales
└── README.md                       # Documentación general
```

---

## ⚡ Reproducibilidad Rápida (Ejecución en Un Solo Paso)

Para ejecutar el pipeline completo de datos, generar todas las figuras a 300 DPI, entrenar los modelos de línea base en validación y compilar el reporte oficial en PDF (máximo 5 páginas):

```bash
# Activar entorno virtual
source .venv/bin/activate

# Ejecutar proceso completo
python run_semana1.py
```
*Tiempo total de ejecución estimado: ~35 segundos.*

---

## 📊 Resumen de Resultados de la Entrega 1 (Validación)

### 1. Definición del Problema y Partición Temporal
* **Horizonte:** $H = 24$ horas hacia adelante.
* **Margen de seguridad:** 24 horas antes del fin de cada período para evitar fugas de información (*data leakage*).
* **Particiones:**
  * **Entrenamiento:** 1 ene 2015 – 30 ago 2015 23:00 ($N = 580,200$, Positivos: $2.02\%$).
  * **Validación:** 1 sep 2015 – 30 oct 2015 23:00 ($N = 144,000$, Positivos: $1.84\%$).
  * **Prueba:** 1 nov 2015 – 30 dic 2015 23:00 ($N = 144,000$, **reservado sin tocar**).

### 2. Desempeño Comparativo de Modelos de Línea Base
Evaluado en el conjunto de validación con desbalance extremo ($1.84\%$ positivos):

| Modelo | PR-AUC | ROC-AUC | Recall (Precisión $\ge 0.80$) | Umbral Óptimo | Macro F1 (Multiclase) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Regresión Logística (Lineal)** | 0.8540 | 0.9977 | 0.8179 | 0.9673 | 0.8028 |
| **Gradient Boosting (HistGBDT)** | **0.9996** | **1.0000** | **1.0000** | **0.0006** | **0.9960** |

### 3. Diagnóstico Multiclase por Componente (HistGBDT)
* **Ninguno (Horas sanas):** Precisión $99.99\%$, Recall $99.99\%$ (sólo 18 falsas alarmas en 141,348 horas).
* **comp1 (Eléctrico):** Precisión $98.03\%$, Recall $100.00\%$, F1 $0.9901$.
* **comp2 (Hidráulico):** Precisión $100.00\%$, Recall $100.00\%$, F1 $1.0000$.
* **comp3 (Mecánico):** Precisión $100.00\%$, Recall $100.00\%$, F1 $1.0000$.
* **comp4 (Rodamientos):** Precisión $98.77\%$, Recall $99.23\%$, F1 $0.9900$.

---

## 🛡️ Mitigación de Fuga de Información (Data Leakage)
1. **Fuga en ventanas móviles:** Todas las agregaciones de sensores (3h y 24h) y conteos de errores son estrictamente causales $[t - W + 1, t]$ sin visión hacia el futuro.
2. **Fuga por registros de mantenimiento:** Se demostró que el $97.6\%$ de los registros de `PdM_maint` coinciden exactamente con fallas (reemplazos correctivos). Se prohibió su uso como indicador en $t$; en su lugar se calculó el tiempo acumulado desde el último reemplazo previo ($t_{\text{maint}} \le t$).
3. **Fuga en normalización:** El escalador `StandardScaler` fue ajustado (*fit*) exclusivamente con datos de entrenamiento.
4. **Fuga en particiones:** Margen de $H=24$h en las fronteras de corte temporal.

