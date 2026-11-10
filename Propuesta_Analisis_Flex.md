# Propuesta de Análisis de Datos y Preguntas para el Socio Formador

**Proyecto:** Optimización de Mantenimiento y Reducción de Tiempos Muertos  
**Fuente:** Hallazgos del Análisis Exploratorio de Datos (EDA_Flex.ipynb)

---

## 1. Resumen y Contexto del Análisis

A partir del análisis exploratorio realizado sobre la base de paros de equipo y tickets de mantenimiento, se identificaron tres patrones clave:
* El 52% de las horas de paro se concentra en máquinas de inyección (IMM) y 20% en robots (juntos explican casi el 72% del tiempo muerto).
* El 29% del tiempo muerto ocurre durante eventos de cambios de modelo, registrando una duración mediana casi del doble que la de un paro correctivo habitual.
* Casi un tercio del tiempo muerto total figura como "Sin clasificar", lo que representa una gran oportunidad para la automatización y calidad de datos.

El análisis propuesto busca anticipar la duración de fallas, clasificar causas faltantes y segmentar las máquinas según su nivel de criticidad.

---

## 2. Estrategia Analítica, Variables y Métodos

### A. Estimación de Duración de Paros (Regresión)
* **Objetivo:** Predecir cuántas horas durará un paro desde el momento en que se genera el reporte, permitiendo planificar mejor la operación y asignar técnicos oportunamente.
* **Variable dependiente:** Tiempo muerto en horas (`Downtime_h`) o tiempo de reparación (`Repair Time`).
* **Variables independientes:** Tipo de equipo (IMM, Robot, Termorregulador), máquina asignada, motivo del reporte (`Reason`), código mayor (`Major Code`), hora del día y día de la semana.
* **Métodos planeados:** Regresión Lineal Múltiple (para evaluar la importancia directa de cada factor) y modelos basados en árboles de decisión (Random Forest o Gradient Boosting) para capturar no linealidades y eventos extremos.

### B. Clasificación de Fallas y Detección de Paros Severos (Clasificación)
* **Objetivo:** Imputar automáticamente el tipo de falla en los paros sin categorizar ("Sin clasificar") y clasificar tempranamente si un paro superará un umbral crítico (mayor a 2 horas).
* **Variable dependiente:** Clase de falla / Código de falla (mecánica, eléctrica, inyección, etc.) o variable binaria de paro crítico (1 = Sí, 0 = No).
* **Variables independientes:** Descripción textual del reporte (longitud del texto y palabras clave), tipo de equipo, máquina, motivo y hora del turno.
* **Métodos planeados:** Árboles de Clasificación, Regresión Logística y modelos de procesamiento de texto sencillo (Naive Bayes o ensamble de árboles con variables derivadas del texto de reporte).

### C. Segmentación de Máquinas Críticas (Clustering)
* **Objetivo:** Agrupar los equipos según su perfil de confiabilidad operativa para diseñar planes de mantenimiento preventivo específicos (ej. equipos con fallas muy frecuentes pero breves vs. equipos que fallan poco pero con paros prolongados).
* **Variables utilizadas (no supervisado):** Número de eventos de falla, tiempo medio entre fallas (MTBF), tiempo medio de reparación (MTTR) y proporción de paros en cambio de modelo.
* **Métodos planeados:** Algoritmo de agrupamiento k-means y análisis jerárquico de conglomerados para clasificar el parque de máquinas en cuadrantes de riesgo (identificando casos prioritarios como Las Vegas, Charros o Consentida).

---

## 3. Preguntas para el Socio Formador

Con el fin de validar el contexto operativo y ajustar los modelos a las prioridades reales de planta:

1. **Paros sin clasificación:** ¿A qué se debe que casi un 30% del tiempo muerto aparezca registrado como "Sin clasificar"? ¿Existe una regla o criterio para que el técnico tipifique o no la causa de la falla?
2. **Impacto en cambio de modelo:** El análisis arrojó que las fallas durante cambio de modelo tienen un tiempo de paro significativamente mayor. ¿Este registro incluye el cambio normal de herramental/molde, o corresponde estrictamente a retrasos e incidentes técnicos ocurridos en ese lapso?
3. **Tiempo real de operación:** Para calcular con exactitud la confiabilidad (MTBF), ¿las máquinas operan de forma continua 24/7, o existen días/turnos sin producción programada según la línea o demanda de producto?
4. **Registro de tiempos de reparación:** Se observó que un 23% de los tickets tienen el tiempo de reparación en cero y la mayoría de los demás valores están redondeados a cuartos de hora. ¿Cómo es el proceso habitual de captura y cierre de tickets por parte del personal de mantenimiento?
5. **Umbral de criticidad en planta:** Desde la perspectiva operativa y de costos de la empresa, ¿a partir de cuántas horas un paro de línea se clasifica como grave o crítico (por ejemplo, paros mayores a 1 hora o mayores a 2 horas)?
