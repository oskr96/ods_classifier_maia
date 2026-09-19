import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from preproc import limpiar_texto, get_ods_name, get_ods_consd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

st.set_page_config(
    page_title="Clasificador ODS: TF-IDF + LSA + LinearSVC",
    page_icon="🤖",
    layout="wide"
)

MODEL_PATH = "ods_classifier.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def predict_texts(_model, texts):
    return _model.predict(texts)


def extract_lsa_info(model):
    return (
        model.named_steps["vectorizador"],
        model.named_steps["lsa"],
        model.named_steps["clasificador"]
    )


# Load model
try:
    model = load_model()
except FileNotFoundError:
    st.error(f"No se encontró '{MODEL_PATH}'.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()


st.title("🤖 Clasificador ODS: TF-IDF + LSA + LinearSVC")
st.caption("TF-IDF → LSA → LinearSVC")

tab_single, tab_eval, tab_lsa, tab_about = st.tabs([
    "🔮 Predicción",
    "📊 Evaluación",
    "🔬 Interpretación LSA",
    "ℹ️ Acerca de"
])


# ─────────────────────────────────────────────
# Predicción
# ─────────────────────────────────────────────
with tab_single:
    st.header("Predecir un texto")
     
    text = st.text_area(
        "Texto de entrada",
        height=160,
        placeholder="Escribe o pega un texto aquí..."
    )

    if st.button("Predecir", type="primary"):
        if not text.strip():
            st.warning("Introduce un texto.")
        else:
            try:
                prediction = model.predict([text])[0]
                name_ods = get_ods_name(prediction)
                st.success(f"Clase predicha: **{prediction}** : **{name_ods}**")
            except Exception as e:
                st.error(f"Error durante la predicción: {e}")
    # Smaller, centered architecture image
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        st.image(
            "architecture.png",
            caption="Arquitectura del modelo",
            width=250
        )


# ─────────────────────────────────────────────
# Evaluación
# ─────────────────────────────────────────────
with tab_eval:
    st.header("Evaluación del modelo")
    st.write(
        "Carga un CSV con una columna de texto y otra de etiquetas reales, usa ; (punto y coma) como separador."
    )
    
    st.write('Recuerda la relación etiqueta->ODS:')
    st.dataframe(get_ods_consd(), use_container_width=True)

    uploaded = st.file_uploader(  "Cargar CSV", type=["csv"], key="evaluation_file")

    if uploaded:
        try:
            data = pd.read_csv(uploaded, sep=';')
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")
            st.stop()

        if data.empty:
            st.warning("El archivo está vacío.")
            st.stop()

        st.dataframe(data.head(10), use_container_width=True)

        columns = list(data.columns)

        text_col = st.selectbox(
            "Columna de texto",
            columns
        )

        label_col = st.selectbox(
            "Columna de etiqueta real",
            columns,
            index=min(1, len(columns) - 1)
        )

        if text_col == label_col:
            st.error("Selecciona columnas diferentes.")

        else:
            clean = st.checkbox(
                "Eliminar filas con valores faltantes",
                value=True
            )

            eval_data = data[[text_col, label_col]].copy()

            if clean:
                eval_data.dropna(
                    subset=[text_col, label_col],
                    inplace=True
                )

            eval_data[text_col] = eval_data[text_col].astype(str)

            st.write(f"Muestras: **{len(eval_data)}**")

            if st.button(
                "Ejecutar evaluación",
                type="primary"
            ):
                if eval_data.empty:
                    st.warning("No hay datos válidos.")

                else:
                    y_true = eval_data[label_col].to_numpy()

                    try:
                        y_pred = predict_texts(
                            model,
                            eval_data[text_col].tolist()
                        )
                    except Exception as e:
                        st.error(
                            f"Error durante la predicción: {e}"
                        )
                        st.stop()

                    labels = np.unique(
                        np.concatenate([
                            y_true,
                            np.asarray(y_pred)
                        ])
                    )

                    metrics = {
                        "Accuracy": accuracy_score(
                            y_true, y_pred
                        ),
                        "Precision": precision_score(
                            y_true,
                            y_pred,
                            average="macro",
                            zero_division=0
                        ),
                        "Recall": recall_score(
                            y_true,
                            y_pred,
                            average="macro",
                            zero_division=0
                        ),
                        "F1": f1_score(
                            y_true,
                            y_pred,
                            average="macro",
                            zero_division=0
                        )
                    }

                    st.session_state["eval"] = (
                        eval_data,
                        label_col,
                        y_true,
                        y_pred,
                        labels,
                        metrics
                    )

            if "eval" in st.session_state:
                (
                    eval_data,
                    label_col,
                    y_true,
                    y_pred,
                    labels,
                    metrics
                ) = st.session_state["eval"]

                st.divider()
                st.subheader("Métricas")

                cols = st.columns(4)

                for col, (name, value) in zip(
                    cols,
                    metrics.items()
                ):
                    col.metric(
                        name,
                        f"{value:.4f}"
                    )

                st.subheader("Reporte de clasificación")

                report = classification_report(
                    y_true,
                    y_pred,
                    zero_division=0,
                    output_dict=True
                )

                st.dataframe(
                    pd.DataFrame(
                        report
                    ).transpose().round(4),
                    use_container_width=True
                )

                st.subheader("Matriz de confusión")

                cm = confusion_matrix(
                    y_true,
                    y_pred,
                    labels=labels
                )

                fig, ax = plt.subplots(
                    figsize=(9, 7)
                )

                ConfusionMatrixDisplay(
                    cm,
                    display_labels=labels,
                ).plot(
                    ax=ax,
                    xticks_rotation="vertical",
                    colorbar=False,
                    cmap="Blues"
                )

                ax.set_title("Matriz de confusión")

                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("Predicciones")

                output = eval_data.copy()
                output["Etiqueta predicha"] = y_pred
                output["Correcta"] = (
                    output[label_col].to_numpy()
                    == y_pred
                )

                st.dataframe(
                    output,
                    use_container_width=True
                )

                st.download_button(
                    "Descargar predicciones CSV",
                    output.to_csv(
                        index=False
                    ).encode("utf-8"),
                    "ods_predictions.csv",
                    "text/csv"
                )


# ─────────────────────────────────────────────
# Interpretación LSA
# ─────────────────────────────────────────────
with tab_lsa:
    st.header("Interpretación de LSA")

    vectorizer, lsa, classifier = extract_lsa_info(model)

    features = vectorizer.get_feature_names_out()
    components = lsa.components_

    topic = st.slider(
        "Seleccionar componente",
        1,
        components.shape[0],
        1
    )

    top_n = st.slider(
        "Número de términos",
        5,
        min(30, len(features)),
        min(10, len(features))
    )

    weights = components[topic - 1]
    idx = np.argsort(weights)[::-1][:top_n]

    terms = pd.DataFrame({
        "Término": features[idx],
        "Peso": weights[idx]
    })

    st.subheader(
        f"Términos principales — Componente {topic}"
    )

    st.dataframe(
        terms.round(5),
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        terms["Término"][::-1],
        terms["Peso"][::-1]
    )

    ax.set_xlabel("Peso")
    ax.set_title(
        f"Componente LSA {topic}"
    )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "Los componentes LSA representan dimensiones latentes "
        "y no necesariamente temas definidos por humanos."
    )

    if st.checkbox("Mostrar todos los pesos"):
        st.dataframe(
            pd.DataFrame(
                components,
                columns=features,
                index=[
                    f"Componente {i + 1}"
                    for i in range(len(components))
                ]
            ).round(5),
            use_container_width=True
        )


# ─────────────────────────────────────────────
# Acerca de
# ─────────────────────────────────────────────
with tab_about:
    st.title("Acerca de")

    st.markdown("""
    ## Clasificador de texto ODS

    Aplicación para clasificación multicategoría de textos
    relacionados con los **Objetivos de Desarrollo Sostenible (ODS)**
    utilizando **TF-IDF + LSA + LinearSVC**.

    ### Funcionalidades

    - 🔮 Predicción de textos
    - 📄 Evaluación de performance haciendo uso de un archivo CSV anotado
    - 🔬 Interpretación de componentes LSA (palabras por topico)
    - 📊 Accuracy, Precision, Recall y F1
    - 🔲 Matriz de confusión
    - 💾 Descarga de predicciones

    ### Modelo

    **Método:** Pipeline: TF-IDF, LSA y clasificadorlinearSVC    
    **Tarea:** Clasificación multicategoría   

    **Versión:** 1.0  
    **Autores:** `Oscar Ferney Gallo Romero` / `Andres Felipe Becerra Puerto`   
    **Universidad de los Andes**  
    **MAIA**
    """)