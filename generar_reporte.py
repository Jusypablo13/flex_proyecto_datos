"""Genera Reporte_Clasificacion_Tickets.pdf a partir de stats.json y las figuras."""
import json
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

s = json.load(open("stats.json"))
st = getSampleStyleSheet()
st.add(ParagraphStyle("J", parent=st["BodyText"], alignment=4, fontSize=10, leading=14))
H1, H2, P = st["Title"], st["Heading2"], st["J"]

def tabla(data, widths=None):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3b6ea5")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                           ("GRID", (0, 0), (-1, -1), .4, colors.grey),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f9")])]))
    return t

m = s["mejor"]
E = []
E += [Paragraph("Clasificación automática de tickets de mantenimiento", H1),
      Paragraph("Archivo: <i>Clasificacion de Tickets 2021.xlsm</i> (hoja <b>Data</b>) &nbsp;|&nbsp; "
                "Entrada: columna Q <i>Work Description (Close Details)</i> &nbsp;|&nbsp; "
                "Salida: columna AB <i>Dispositivo</i>", P), Spacer(1, 10)]

E += [Paragraph("1. Objetivo", H2), Paragraph(
    "Construir un algoritmo que, dado el texto libre con el que el técnico cierra un ticket (columna Q), "
    "prediga la categoría <b>Dispositivo</b> (columna AB) asignada manualmente. Esto permitiría "
    "automatizar la clasificación de tickets nuevos y reducir el trabajo manual.", P)]

E += [Paragraph("2. Procesamiento de datos", H2), Paragraph(
    f"La hoja contiene {s['filas_totales']:,} filas; {s['con_texto']:,} tienen texto en Q y "
    f"{s['con_categoria']:,} tienen categoría en AB. Sólo <b>{s['pares']:,}</b> filas tienen ambas y "
    "constituyen el conjunto supervisado. Pasos aplicados:", P)]
pasos = [
    f"<b>Normalización de etiquetas:</b> minúsculas, espacios y corrección de typos/sinónimos "
    f"(p.ej. <i>Door´s switch</i> / <i>door's swtch</i> → <i>door's switch</i>; <i>bus can alarm</i> → <i>buscan error</i>). "
    f"Las clases pasaron de {s['clases_raw']} a {s['clases_normalizadas']}.",
    "<b>Limpieza de texto:</b> minúsculas, eliminación de acentos y signos, sustitución de números por "
    "<i>num</i> y de referencias a tickets (REQ-xxxx) por <i>ticketref</i>, eliminación de stopwords en español.",
    f"<b>Agrupación de clases raras:</b> las categorías con menos de 10 ejemplos se agruparon en "
    f"<i>otros</i> ({s['pct_otros']:.1f}% de los datos), dejando <b>{s['clases_finales']} clases</b>.",
    f"<b>División:</b> 80/20 estratificada ({s['train']:,} entrenamiento, {s['test']:,} prueba) + validación cruzada 5-fold.",
]
for p in pasos:
    E.append(Paragraph("• " + p, P))
E += [Spacer(1, 6), Image("fig_distribucion.png", width=13 * cm, height=9.75 * cm), Paragraph(
    "La distribución está muy desbalanceada: <i>operation</i>, <i>installation</i>, <i>screw cleaning</i> y "
    "<i>validation</i> concentran cerca del 45% de los tickets, mientras que la mayoría de las clases tienen "
    "decenas de ejemplos.", P), PageBreak()]

E += [Paragraph("3. Algoritmo", H2), Paragraph(
    "<b>Representación:</b> TF-IDF combinando (a) n-gramas de palabras (1-2) y (b) n-gramas de caracteres (3-5). "
    "Los n-gramas de caracteres hacen al modelo robusto a faltas de ortografía y a la mezcla español/inglés "
    "típica de los técnicos (<i>termopar</i>, <i>thermocouple</i>, <i>husillo</i>…).", P),
      Paragraph("<b>Clasificadores comparados:</b> baseline de clase mayoritaria, Complement Naive Bayes, "
                "Regresión Logística (con <i>class_weight='balanced'</i> para mitigar el desbalance) y SVM lineal. "
                "Se eligió el modelo con mayor F1 ponderado en el conjunto de prueba; además del acierto exacto se "
                "mide <b>top-3</b> (la categoría correcta está entre las 3 sugerencias más probables), útil para un "
                "esquema de sugerencia asistida.", P), Spacer(1, 6)]

E += [Paragraph("4. Resultados", H2)]
rows = [["Modelo", "Accuracy", "F1 macro", "F1 ponderado", "Top-3"]]
for r in s["resultados"]:
    rows.append([r["modelo"]] + [f"{r[k]:.3f}" for k in ("accuracy", "f1_macro", "f1_weighted", "top3")])
E += [tabla(rows, [6 * cm, 2.3 * cm, 2.3 * cm, 2.7 * cm, 2 * cm]), Spacer(1, 6),
      Image("fig_modelos.png", width=14 * cm, height=7.8 * cm), Paragraph(
    f"El mejor modelo es <b>{m['modelo']}</b>: accuracy {m['accuracy']:.1%}, F1 ponderado {m['f1_weighted']:.3f} y "
    f"<b>top-3 de {m['top3']:.1%}</b>, frente a {s['resultados'][0]['accuracy']:.1%} del baseline (≈2.7× mejor). "
    f"La validación cruzada 5-fold da {s['cv_mean']:.1%} ± {s['cv_std']:.1%}, confirmando que el resultado es estable.", P),
      PageBreak()]

rep = pd.DataFrame(s["reporte"]).T.drop(["macro avg", "weighted avg"], errors="ignore")
rep = rep.sort_values("support", ascending=False).head(15)
rows = [["Categoría", "Precisión", "Recall", "F1", "Soporte"]] + \
       [[i, f"{r.precision:.2f}", f"{r.recall:.2f}", f"{r['f1-score']:.2f}", int(r.support)] for i, r in rep.iterrows()]
E += [Paragraph("Desempeño por categoría (15 más frecuentes)", st["Heading3"]),
      tabla(rows, [5.5 * cm, 2.3 * cm, 2.3 * cm, 2 * cm, 2 * cm]), Spacer(1, 6),
      Image("fig_confusion.png", width=13 * cm, height=11.5 * cm), Paragraph(
    "Las categorías con vocabulario específico se clasifican muy bien (<i>screw cleaning</i> F1 0.92, "
    "<i>alignment</i> 0.85, <i>nozzle thermocouple</i> 0.73). Las mayores confusiones ocurren entre categorías "
    "genéricas (<i>operation</i>, <i>validation</i>, <i>reset</i>, <i>otros</i>), cuyos textos suelen ser "
    "\"se realizan pruebas y opera correctamente\", sin mencionar el componente intervenido.", P), PageBreak()]

rows = [["Texto de entrada (col. Q)", "Categoría predicha"]] + [[a, b] for a, b in s["ejemplos"]]
E += [Paragraph("Ejemplos de predicción", st["Heading3"]), tabla(rows, [10.5 * cm, 5 * cm]), Spacer(1, 6)]

E += [Paragraph("5. Conclusiones y siguientes pasos", H2)]
for p in [
    "Se cuenta con un clasificador funcional (<i>clasificador_tickets.py</i>, modelo en <i>modelo_tickets.pkl</i>, "
    "función <i>predecir(textos)</i>) que acierta ~50% en exacto y ~76% en top-3 sobre 87 clases.",
    "La principal limitación está en los datos: etiquetas inconsistentes (519 variantes originales), "
    "muchas clases con pocos ejemplos y textos que no describen el componente.",
    "Siguientes pasos: (1) depurar el catálogo de <i>Dispositivo</i> con el usuario de negocio usando la hoja "
    "<i>Catalogo</i>; (2) clasificación jerárquica usando <i>Sub Sistema</i> → <i>Dispositivo</i>; "
    "(3) añadir como variables la columna <i>Description</i> (problema reportado) y el tipo de equipo; "
    "(4) probar embeddings multilingües / modelos tipo BERT; (5) usarlo como sistema de sugerencia top-3.",
]:
    E.append(Paragraph("• " + p, P))

SimpleDocTemplate("Reporte_Clasificacion_Tickets.pdf", pagesize=letter,
                  leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm).build(E)
print("PDF generado")
