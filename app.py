import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

def apply_kmeans(x_data, k=3):
    """
    Aplica el algoritmo K-Means a los datos proporcionados.

    Args:
    x_data (array-like): Datos de entrada para el clustering.
    k (int): Numero de clusters. Por defecto es 3.

    Returns:
    labels (array): Etiquetas de cluster asignadas a cada punto de datos.
    inertia (float): Inercia del modelo K-Means.
    groups_sizes (array): Tamaños de cada grupo/clúster.
    """
    model = KMeans(n_clusters=k,    # numero de clusterss
                   n_init='auto',   # numero de inicializaciones, 'auto' es para que sklearn elija el mejor valor
                   random_state=42  # semilla para reproducibilidad, es para que los resultados sean consistentes,
                                    # cada vez que se ejecute el codigo
                   )
    # la funcion fit ajusta el modelo a los datos, es decir, encuentra los centroides de los clusters
    model.fit(x_data)
    # .labels_ contiene las etiquetas asignadas a cada punto de datos, indicando a que cluster pertenece
    labels = model.labels_
    # .inertia_ es una medida de como de compactos son los clusters, es la suma de las distancias cuadradas
    # entre cada punto y el centroide de su cluster
    inertia = model.inertia_
    # np.bincount cuenta el numero de ocurrencias de cada etiqueta en "labels"
    # esta es una forma eficiente de obtener el tamanio de cada cluster
    groups_sizes = np.bincount(labels)
    return labels, inertia, groups_sizes

def apply_linear_regression(data : pd.DataFrame, features_x : list, target_y: str) -> None:
    """
    Entrena y evalúa un modelo de Regresión Lineal Múltiple.

    Esta función prepara los datos (codificación one-hot), divide el conjunto
    en entrenamiento y prueba, entrena el modelo de regresión lineal y calcula
    las métricas de evaluación (R2 y RMSE).

    Args:
        data (pd.DataFrame): El DataFrame completo que contiene los datos a analizar.
        features_x (list): Lista de cadenas con los nombres de las columnas
                            seleccionadas como variables independientes (X).
        target_y (str): Nombre de la columna seleccionada como variable
                        dependiente (Y), que debe ser cuantitativa.

    Returns:
        r2 (float): El R2 Score del modelo entrenado.
        rmse (float): La Raiz del Error Cuadratico Medio del modelo entrenado.
        feature_names (list): Nombres de las caracteristicas (X) utilizadas despues del One-Hot Encoding.
        coefficients (array): Los coeficientes del modelo para cada caracteristica.
        intercept (float): El intercepto (ordenada al origen) del modelo.
    """

    # Extraemos la columna X y Y.
    selected_columns = features_x + [target_y]
    # Creamos una copia del dataframe tomando solo las columnas seleccionadas.
    df_model = data[selected_columns].copy()

    # === Aplicamos One-Hot Encoding ===
    # Esto se utiliza para convertir variables categóricas en variables numéricas, que son más fáciles de trabajar ===
    # Por ejemplo: Antes -> Categoria Sexo: [Hombre, Mujer] | Después -> Categoria Sexo: [0, 1].
    df_model = pd.get_dummies(df_model, drop_first=True)

    # Asegurarnos de que target_y siga existiendo despues de get_dummies
    if target_y not in df_model.columns:
        # Esto es un parche por si target_y era una de las columnas eliminadas por drop_first
        # Idealmente, target_y siempre es numerica y no se ve afectada
        # Si se ve afectada, la logica de get_dummies necesita ser mas robusta
        st.error(
            f"Error: La variable objetivo '{target_y}' se vio afectada por la codificación. Asegúrese de que sea numérica.")
        return None, None, None, None

    # === Tenemos que separar las caracteristicas [X], del objetivo [Y] ===

    # "Y" es la variable que queremos predecir
    Y = df_model[target_y]

    # "X" Son todas las demas columnas, excepto Y
    X = df_model.drop(columns=target_y)

    # === Dividimos los datos en Entrenamiento y prueba ===

    # Nota: Aunque solo trabajamos con las primeras 100 filas (80 para entrenamiento, 20 para prueba),
    # se mantiene la proporción estándar del 20% para el conjunto de prueba.
    # En un entorno de producción, se usaría el DataFrame completo.
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        # Se usa el 20% de los datos para el entrenamiento.
        test_size=0.2,
        # Semilla para la division de los datos
        random_state=42
    )

    # === Aqui entrenamos el modelo :)
    model = LinearRegression()

    # Pasamos los datosd del entrenamiento
    model.fit(X_train, Y_train)

    # Hacemos la prediccion sobre el conjunto de prueba
    Y_pred = model.predict(X_test)

    # === Evaluacion del modelo ===

    # Calculamos el R2 Score
    # Mas que nada, mide como se ajustan los datos del 0 al 1 (1 significa perfecto)
    r2 = r2_score(Y_test, Y_pred)

    # Calculamos el Error Cuadratico Medio MSE
    # Mide la magnitud promedio del error
    mse = mean_squared_error(Y_test, Y_pred)

    # Calculamos la raiz del Error Cuadratico Medio RMSE
    # Trabaja los mismos valores del objetivo
    rmse = np.sqrt(mse)

    # === Prueba de resultados en consola ===
    #print("\n--- Resultados de la Regresión Lineal Múltiple ---")
    #print(f"Variables de Entrada (X) usadas: {list(X.columns)}")
    #print(f"Variable a Predecir (Y): {target_y}")
    #print("-" * 50)
    #print(f"R2 Score: {r2:.4f}")
    #print(f"RMSE (Error): {rmse:.2f}")
    return r2, rmse, X.columns.tolist(), model.coef_, model.intercept_, Y_test, Y_pred


# === Funciones para Graficar ===
def plot_kmeans_results(data: pd.DataFrame, features_x: list, labels: np.ndarray):
    """
    Crea un gráfico de dispersión de K-Means si hay 2 o 3 variables.

    Args:
        data (pd.DataFrame): El DataFrame completo que contiene los datos a analizar.
        features_x (list): Lista de cadenas con los nombres de las columnas seleccionadas como variables
            independientes (X).
        labels (np.ndarray): Etiquetas de clúster asignadas a cada punto de datos.
    """
    # agregamos las etiquetas de cluster al dataframe para que Plotly pueda usarlas
    data_plot = data[features_x].copy()
    data_plot['Cluster'] = labels.astype(str)  # Convertir a string para colores categoricos

    if len(features_x) == 2:
        # Grafico 2D
        fig = px.scatter(
            data_plot,
            x=features_x[0],
            y=features_x[1],
            color='Cluster',
            title="Visualización de Clústeres (K-Means)"
        )
        st.plotly_chart(fig, width='stretch')
    elif len(features_x) == 3:
        # Grafico 3D
        fig = px.scatter_3d(
            data_plot,
            x=features_x[0],
            y=features_x[1],
            z=features_x[2],
            color='Cluster',
            title="Visualización de Clústeres (K-Means) 3D"
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Seleccione 2 o 3 variables numéricas en (X) para generar un gráfico de dispersión.")


# Grafica de la Regresion Lineal
def plot_linear_regression_results(Y_test: pd.Series, Y_pred: np.ndarray, target_y: str):
    """
    Crea un grafico de dispersion de los valores Reales vs Predichos para Regresion Lineal.
    """
    
    # Creamos un dataframe para el plot
    df_plot = pd.DataFrame({
        f'Valores Reales de {target_y}': Y_test,
        f'Valores Predichos de {target_y}': Y_pred
    })

    # Creamos la fig
    fig = px.scatter(
        df_plot,
        x = f'Valores Reales de {target_y}',
        y = f'Valores Predichos de {target_y}',
        title = f"Regresion Lineal: Valores Reales vs Predichos de '{target_y}'",
        labels = {'x': 'Valor Real', 'y': 'Valor Predicho'}
    )

    # Linea diagonal de la funcion
    max_val = max(Y_test.max(), Y_pred.max())
    min_val = min(Y_test.min(), Y_pred.min())
    fig.add_shape(
        type="line",
        x0=min_val, y0=min_val,
        x1=max_val, y1=max_val,
        line=dict(color="red", width=2, dash="dash"),
        name='Ajuste Perfecto(Y = X)'
    )

    st.plotly_chart(fig, width='stretch')


# === Funciones de Visualizacion ===
def display_kmeans_results(data: pd.DataFrame, features_x: list):
    """
    Prepara los datos, ejecuta K-Means y muestra los resultados en Streamlit.
    """
    st.subheader("Resultados de K-Means")

    # preparar los datos
    x_data = data[features_x].copy()

    # K-Means solo funciona con datos numericos. Filtramos.
    x_data_numeric = x_data.select_dtypes(include=np.number)

    # Validar
    if x_data_numeric.empty:
        st.error(
            "Error: K-Means solo puede ejecutarse sobre variables numéricas. Por favor, seleccione columnas con /"
            "números.")
        return

    if x_data_numeric.shape[1] < len(features_x):
        st.warning("Advertencia: Se ignoraron algunas columnas no numéricas seleccionadas.")

    # ejecutar el modelo
    labels, inertia, groups_sizes = apply_kmeans(x_data_numeric, k=3)

    # mostrar resultados en la interfaz
    st.write("El modelo ha clasificado los datos en 3 grupos:")

    # usamos st.metric para un buen impacto visual
    st.metric(label="Inercia Total (Suma de distancias cuadradas)", value=f"{inertia:.2f}")

    st.write("Tamaño de cada clúster:")
    # creamos un DataFrame para mostrar los tamanios en una tabla
    df_sizes = pd.DataFrame({
        'Clúster': [f"Clúster {i}" for i in range(len(groups_sizes))],
        'Número de Registros': groups_sizes
    })
    st.dataframe(df_sizes)

    # graficar los resultados
    plot_kmeans_results(data, features_x, labels)


def display_linear_regression_results(r2, rmse, featured_used, coefs, intercept, Y_test, Y_pred, target_y):
    """
    Muestra las metricas, coeficientes y la visualizacion de la Regresion Lineal en StreamLit.
    """
    st.subheader("Resultados de Regresion Lineal Multiple")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="R² Score (Bondad de Ajuste)", value=f"{r2:.4}", help="Cercano a 1 es mejor.")
    with col2:
        st.metric(label=f"RMSE (Error de Prediccion en {target_y})", value=f"{rmse:.2f}", help="Valor de error promedio, en las mismas unidades que Y.")

    st.write("---")

    # Mostrar el coeficiente del modelo
    st.write("### Coeficientes del modelo")
    # Dataframe para mostrar
    df_coefs = pd.DataFrame({
        'Variable': featured_used,
        'Coeficiente': coefs
    })
    st.dataframe(df_coefs, width='stretch')
    st.write(f"**Intercepto (Ordenada al origen):** '{intercept:.4f}'")

    st.write("---")

    # Visualizacion
    st.write("### Visualizacion de Predicciones")
    plot_linear_regression_results(Y_test, Y_pred, target_y)


def apply_logistic_regression(X: pd.DataFrame, Y: pd.Series):
    """
    Entrena un modelo de Regresión Logística para un problema de clasificación binaria.
    Esta función asume que los datos ya han sido preparados.
    y que la variable objetivo (Y) es binaria (0 o 1).
    Args:
        X (pd.DataFrame): Variables independientes (features).
        Y (pd.Series): Variable dependiente (target), debe ser binaria.
    Returns:
        tuple: Tupla que contiene:
            - coefficients (np.ndarray): Coeficientes (pesos) del modelo.
            - intercept (float): Intercepto del modelo.
            - feature_names (list): Nombres de las caracteristicas (X) utilizadas.
    """
    
    # === Dividimos los datos de entrenamiento y prueba ===
    # Tengo entendido que es buena practica dividir los datos, incluso si qui no se evaluan
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size = 0.2,
        random_state = 42,
        stratify = Y
    )

    # === Escalamos los datos ===
    # Sirve para converger mas rapido
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # === Creamos y entrenamos el modelo
    # Liblinear sirve para conjuntos pequenos
    model = LogisticRegression(solver="liblinear", random_state=42)

    # Entrenamos el modelo
    model.fit(X_train_scaled, Y_train)
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]

    return coefficients, intercept, X_train.columns.tolist()


def main():
    st.header("Cargar Archivo")
    file = st.file_uploader("Elige un archivo CSV o XLSX", type=["csv", "xlsx"])

    if file is not None:
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file)
            st.success("¡Archivo cargado exitosamente!")
            # --- Vista previa de los datos ---
            df_display = df.head(100)
            st.dataframe(df_display)

            columns = df.columns.tolist()
            selected_algorithm = st.selectbox(
                "Seleccione el algoritmo a ejecutar:",
                ["Regresión Lineal Múltiple", "Regresión Logística Binaria", "K-Means"]
            )
            variables_x = st.multiselect(
                "Seleccione las variables de atributos (x):",
                options=columns,
                help="Estas son las variables que el modelo usará para predecir."
            )

            variable_y = None
            if selected_algorithm != "K-Means":
                # filtramos las opciones para la variable "y" de modo que
                # no se pueda seleccionar una variable que ya este en "x"
                options_y = [col for col in columns if col not in variables_x]

                # === La variable "y", debe ser cuantitativa. Hay que mejorar esto en el formulario ===
                variable_y = st.selectbox(
                    "Seleccione la variable de la clase (y):",
                    options=options_y,
                    help="Esta es la variable que el modelo intentará predecir."
                )
                # si el usuario selecciona k-means
            if st.button("Ejecutar Análisis"):
                if selected_algorithm == "K-Means":
                    if not variables_x:
                        st.warning("Por favor, seleccione al menos una variable de atributo (x) para K-Means.")
                    else:
                        display_kmeans_results(df_display, variables_x)

                # Si el usuario selecciona regresion lineal
                elif selected_algorithm == "Regresión Lineal Múltiple": 
                    if variables_x and variable_y:
                        r2, rmse, features_used, coefs, intercept, Y_test, Y_pred = apply_linear_regression(
                            df_display, variables_x, variable_y
                        )

                        # Validacion, por si hay errores
                        if r2 is not None:
                                display_linear_regression_results(
                                    r2, rmse, features_used, coefs, intercept, 
                                    Y_test, Y_pred, variable_y
                                )
                        else:
                            st.error("No se pudo ejecutar la Regresión Lineal Múltiple. Revise la consola para detalles.")


                # Si el ususario selecciona regresion logistica binaria
                elif selected_algorithm == "Regresión Logística Binaria":
                    pass

        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
    
if __name__ == "__main__":
    main()
