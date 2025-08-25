
import joblib
import streamlit as st
import pandas as pd

# --- Configuración de la Página ---
# Esto debe ser lo primero que se ejecute en el script.
st.set_page_config(
    page_title="Predictor de Rendimiento de Destilación",
    page_icon="🧪",
    layout="wide"
)

# --- Carga del Modelo ---
# Usamos @st.cache_resource para que el modelo se cargue solo una vez y se mantenga en memoria,
# lo que hace que la aplicación sea mucho más rápida.
@st.cache_resource
def load_model(model_path):
    """Carga el modelo entrenado desde un archivo .joblib."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo del modelo en {model_path}. Asegúrate de que el archivo del modelo esté en el directorio correcto.")
        return None

# Cargamos nuestro modelo campeón. Streamlit buscará en la ruta 'modelo_xgboost_final.joblib'.
model = load_model('modeloproyecto.joblib')

# --- Barra Lateral para las Entradas del Usuario ---
with st.sidebar:
    st.header("⚙️ Parámetros de Entrada")
    st.markdown("""
    Ajusta los deslizadores para que coincidan con los parámetros operativos de la columna de destilación.
    """)

    # Slider para el caudal de alimentación
    AmineFlow = st.slider(
        label='Flujo de Amina (kg/min)',
        min_value=241,
        max_value=740,
        value=300, # Valor inicial
        step=1
    )
    st.caption("Representa el flujo de amina en el proceso.")

    # Slider para la temperatura
    AirFlow = st.slider(
        label='Flujo de Aire Columna 1(kg/min)',
        min_value=175,
        max_value=372,
        value=180,
        step=1
    )
    st.caption("Importante para el proceso de transferencia de masa (burbujeo)")

    # Slider para la diferencia de presión
    Iron = st.slider(
        label='Hierro Concentrado (%)',
        min_value=62,
        max_value=69,
        value=63,
        step=1
    )
    st.caption("Condiciona parametros de selectividad")

# --- Contenido de la Página Principal ---
st.title("🧪 Predictor de Rendimiento de Proceso de Flotacion")
st.markdown("""
¡Bienvenido! Esta aplicación utiliza un modelo de machine learning para predecir el porcentaje de Silica Concentrada en una actividad de Flotación minera
**Esta herramienta puede ayudar a los ingenieros de procesos y operadores a:**
- **Optimizar** las condiciones de operación para obtener el máximo rendimiento.
- **Predecir** el impacto de los cambios en el proceso antes de implementarlos.
- **Solucionar** problemas potenciales simulando diferentes escenarios.
""")

# --- Lógica de Predicción ---
# Solo intentamos predecir si el modelo se ha cargado correctamente.
if model is not None:
    # El botón principal que el usuario presionará para obtener un resultado.
    if st.button('🚀 Predecir Rendimiento', type="primary"):
        # Creamos un DataFrame de pandas con las entradas del usuario.
        # ¡Es crucial que los nombres de las columnas coincidan exactamente con los que el modelo espera!
        df_input = pd.DataFrame({
            'FlujoAmina': [AmineFlow],
            'FlujoAireColumna1': [AirFlow],
            'Hierro Concentrado': [Iron]
        })

        # Hacemos la predicción
        try:
            prediction_value = model.predict(df_input)
            st.subheader("📈 Resultado de la Predicción")
            # Mostramos el resultado en un cuadro de éxito, formateado a dos decimales.
            st.success(f"**Rendimiento Predicho:** `{prediction_value[0]:.2f}%`")
            st.info("Este valor representa el porcentaje estimado del producto deseado que se recuperará.")
        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción: {e}")
else:
    st.warning("El modelo no pudo ser cargado. Por favor, verifica la ruta del archivo del modelo.")

st.divider()

# --- Sección de Explicación ---
with st.expander("ℹ️ Sobre la Aplicación"):
    st.markdown("""
    **¿Cómo funciona?**

    1.  **Datos de Entrada:** Proporcionas los parámetros operativos clave usando los deslizadores en la barra lateral.
    2.  **Predicción:** El modelo de machine learning pre-entrenado recibe estas entradas y las analiza basándose en los patrones que aprendió de datos históricos.
    3.  **Resultado:** La aplicación muestra el rendimiento final predicho como un porcentaje.

    **Detalles del Modelo:**

    * **Tipo de Modelo:** `Regression Model` (XGBoost Optimizado)
    * **Propósito:** Predecir el valor continuo del rendimiento de la destilación.
    * **Características Usadas:** Flujo de Amina, Flujo de Aire en la Columna 1 y Porcentaje de Hierro Concentrado.
    """)
