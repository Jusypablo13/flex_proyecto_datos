"""Clasificador de tickets: texto (col. Q 'Work Description') -> categoría (col. AB 'Dispositivo').

Uso:
    python clasificador_tickets.py            # entrena, evalúa y genera figuras + reporte PDF
    python -c "from clasificador_tickets import predecir; print(predecir(['se cambia termopar de nariz']))"
"""
import re, unicodedata, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ARCHIVO = "Clasificacion de Tickets 2021.xlsm"
HOJA = "Data"
COL_Q, COL_AB = 16, 27          # Q = índice 16, AB = índice 27 (base 0)
MIN_EJEMPLOS = 10               # clases con menos ejemplos -> "otros"
SEED = 42

STOP = set("""de la el en los las se y a con por para del al que un una es lo le su sus o
fue son ya no si sin como mas pero ser esta este esto estos se realiza realizo realizan""".split())

# Correcciones de etiquetas detectadas en el análisis (typos / sinónimos)
MAPEO_ETIQUETAS = {
    "door´s switch": "door's switch", "door's swtch": "door's switch",
    "air leake": "air leak", "hidraulic unit": "hydraulic unit",
    "regeneration blower ": "regeneration blower", "autotune": "autotuning",
    "bus can error": "buscan error", "bus can alarm": "buscan error",
    "resine leake": "resine leak", "material leake": "resine leak",
    "alaignment": "alignment", "teach base": "teach pendant base",
}


def limpiar_texto(t: str) -> str:
    t = str(t).lower()
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    t = re.sub(r"req-?\d+|ritm\d+|inc\d+", " ticketref ", t)   # referencias a tickets
    t = re.sub(r"\d+", " num ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return " ".join(w for w in t.split() if w not in STOP)


def limpiar_etiqueta(e: str) -> str:
    e = re.sub(r"\s+", " ", str(e).strip().lower())
    return MAPEO_ETIQUETAS.get(e, e)


def cargar():
    df = pd.read_excel(ARCHIVO, sheet_name=HOJA)
    q, ab = df.columns[COL_Q], df.columns[COL_AB]
    datos = df[[q, ab]].dropna().rename(columns={q: "texto", ab: "categoria_raw"})
    stats = {"filas_totales": len(df), "con_texto": int(df[q].notna().sum()),
             "con_categoria": int(df[ab].notna().sum()), "pares": len(datos),
             "clases_raw": int(datos["categoria_raw"].nunique())}
    datos["texto_limpio"] = datos["texto"].map(limpiar_texto)
    datos["categoria"] = datos["categoria_raw"].map(limpiar_etiqueta)
    stats["clases_normalizadas"] = int(datos["categoria"].nunique())
    datos = datos[datos["texto_limpio"].str.len() > 0]
    conteo = datos["categoria"].value_counts()
    validas = conteo[conteo >= MIN_EJEMPLOS].index
    datos["categoria_final"] = np.where(datos["categoria"].isin(validas), datos["categoria"], "otros")
    stats["clases_finales"] = int(datos["categoria_final"].nunique())
    stats["pct_otros"] = float((datos["categoria_final"] == "otros").mean() * 100)
    return datos, stats


def vectorizador():
    return FeatureUnion([
        ("palabras", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("caracteres", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])


def modelos():
    return {
        "Baseline (clase mayoritaria)": DummyClassifier(strategy="most_frequent"),
        "Complement Naive Bayes": ComplementNB(alpha=0.3),
        "Regresión Logística": LogisticRegression(C=10, max_iter=2000, class_weight="balanced"),
        "SVM lineal (LinearSVC)": LinearSVC(C=0.5, class_weight="balanced"),
    }


def topk(clf, X, y, k=3):
    s = clf.decision_function(X) if hasattr(clf, "decision_function") else clf.predict_proba(X)
    idx = np.argsort(-s, axis=1)[:, :k]
    return float(np.mean([y[i] in clf.classes_[idx[i]] for i in range(len(y))]))


def main():
    datos, stats = cargar()
    X, y = datos["texto_limpio"].values, datos["categoria_final"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    stats.update(train=len(Xtr), test=len(Xte))

    resultados, mejores = [], None
    for nombre, m in modelos().items():
        pipe = Pipeline([("tfidf", vectorizador()), ("clf", m)])
        pipe.fit(Xtr, ytr)
        p = pipe.predict(Xte)
        r = {"modelo": nombre, "accuracy": accuracy_score(yte, p),
             "f1_macro": f1_score(yte, p, average="macro", zero_division=0),
             "f1_weighted": f1_score(yte, p, average="weighted", zero_division=0)}
        r["top3"] = topk(pipe.named_steps["clf"], pipe.named_steps["tfidf"].transform(Xte), yte) \
            if nombre.startswith(("Reg", "SVM", "Comp")) else r["accuracy"]
        resultados.append(r)
        print(r)
        if mejores is None or r["f1_weighted"] > mejores[1]["f1_weighted"]:
            mejores = (pipe, r, p)

    pipe, rbest, pred = mejores
    # Validación cruzada 5-fold del mejor modelo
    cv = cross_val_score(Pipeline([("tfidf", vectorizador()), ("clf", pipe.named_steps["clf"])]),
                         X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                         scoring="accuracy", n_jobs=-1)
    stats["cv_mean"], stats["cv_std"] = float(cv.mean()), float(cv.std())

    rep = classification_report(yte, pred, output_dict=True, zero_division=0)
    pd.DataFrame(resultados).to_csv("resultados_modelos.csv", index=False)
    pd.DataFrame(rep).T.to_csv("reporte_por_clase.csv")

    # ---------- Figuras ----------
    top = datos["categoria_final"].value_counts().head(20)[::-1]
    plt.figure(figsize=(8, 6)); plt.barh(top.index, top.values, color="#3b6ea5")
    plt.title("Top 20 categorías (Dispositivo) tras normalización"); plt.xlabel("Tickets")
    plt.tight_layout(); plt.savefig("fig_distribucion.png", dpi=150); plt.close()

    dfr = pd.DataFrame(resultados).set_index("modelo")[["accuracy", "f1_macro", "f1_weighted", "top3"]]
    dfr.plot.bar(figsize=(9, 5), rot=15, color=["#3b6ea5", "#e07b39", "#5aa469", "#9b59b6"])
    plt.title("Comparación de modelos (conjunto de prueba)"); plt.ylim(0, 1); plt.tight_layout()
    plt.savefig("fig_modelos.png", dpi=150); plt.close()

    clases = list(datos["categoria_final"].value_counts().head(15).index)
    cm = confusion_matrix(yte, pred, labels=clases, normalize="true")
    plt.figure(figsize=(9, 8)); plt.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    plt.xticks(range(len(clases)), clases, rotation=75, fontsize=8); plt.yticks(range(len(clases)), clases, fontsize=8)
    for i in range(len(clases)):
        for j in range(len(clases)):
            if cm[i, j] >= 0.05:
                plt.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                         color="white" if cm[i, j] > .5 else "black")
    plt.title(f"Matriz de confusión normalizada (15 clases más frecuentes)\n{rbest['modelo']}")
    plt.colorbar(fraction=0.046); plt.tight_layout(); plt.savefig("fig_confusion.png", dpi=150); plt.close()

    # Ejemplos de predicción
    ejemplos = ["se realiza limpieza de husillo con purga", "se cambia termopar de nariz",
                "se reemplaza manguera de aire dañada", "se realizan pruebas funcionales y opera correctamente",
                "se cambia resistencia de nariz", "se reinicia equipo por error de comunicacion can"]
    stats["ejemplos"] = list(zip(ejemplos, pipe.predict([limpiar_texto(e) for e in ejemplos])))

    # Reentrenar con todos los datos y guardar
    pipe.fit(X, y)
    with open("modelo_tickets.pkl", "wb") as f:
        pickle.dump(pipe, f)
    stats["mejor"] = rbest
    stats["resultados"] = resultados
    stats["reporte"] = {k: v for k, v in rep.items() if isinstance(v, dict)}
    with open("stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    print(json.dumps({k: v for k, v in stats.items() if k != "reporte"}, indent=1, default=str))


def predecir(textos):
    with open("modelo_tickets.pkl", "rb") as f:
        pipe = pickle.load(f)
    return list(pipe.predict([limpiar_texto(t) for t in textos]))


if __name__ == "__main__":
    main()
