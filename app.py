import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def main():
    st.header("Cargar Archivo")
    file = st.file_uploader("Elige un archivo CSV o XLSX", type=["csv", "xlsx"])

    df = None

    if file is not None:
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file)
            st.success("¡Archivo cargado exitosamente!")
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
                # we filter the options for ‘Y’ so that a variable that is already in ‘X’ cannot be selected
                options_y = [col for col in columns if col not in variables_x]

                # === La variable "y", debe ser cuantitativa. Hay que mejorar esto en el formulario ===
                variable_y = st.selectbox(
                    "Seleccione la variable de la clase (y):",
                    options=options_y,
                    help="Esta es la variable que el modelo intentará predecir."
                )

            # If user select linear regression.
            if selected_algorithm == "Regresión Lineal Múltiple" and variables_x and variable_y:
                resultados = apply_linear_regression(df_display, variables_x, variable_y)

                # Toda esta seccion es para graficar en streamlit
                if resultados:
                    st.subheader("Resultados del Modelo de Regresión Lineal")
                    st.write(f"Variables usadas: **{', '.join(variables_x)}**")
                    st.write(f"Variable objetivo: **{variable_y}**")
                    
                    # 2. Mostrar las métricas con st.metric
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'R2_Score' in resultados:
                            st.metric("R2 Score", f"{resultados['R2_Score']:.4f}", help="Indica qué porcentaje de la varianza en Y es explicado por X.")
                    with col2:
                        if 'RMSE' in resultados:
                            st.metric("RMSE (Error)", f"${resultados['RMSE']:.2f}", help="El error promedio de la predicción, en las unidades de la variable Y.")

                    # 3. Mostrar el gráfico con st.pyplot
                    st.markdown("---")
                    st.markdown("#### Gráfico de Evaluación (Valores Reales vs. Predichos)")
                    st.pyplot(resultados["Plot_Figure"]) # Renderiza la figura de Matplotlib

        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")

def apply_linear_regression(data : pd.DataFrame, features_x : list, target_y: str):
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
        dict: Un diccionario conteniendo las métricas y la figura de visualización.
            - R2_Score (float): El coeficiente de determinación. Mide la proporción de la varianza en Y explicada por X.
            - RMSE (float): La raíz del error cuadrático medio. Mide el error promedio en las unidades de Y.
            - Coefficients (np.ndarray): Los coeficientes (pesos) asignados por el modelo a cada característica X.
            - Plot_Figure (matplotlib.figure.Figure): Objeto figura con el gráfico de evaluación Real vs. Predicho.
    """

    # Extraemos la columna X y Y.
    selected_columns = features_x + [target_y]
    # Creamos una copia del dataframe tomando solo las columnas seleccionadas.
    df_model = data[selected_columns].copy()

    # === Aplicamos One-Hot Encoding ===
    # Esto se utiliza para convertir variables categóricas en variables numéricas, que son más fáciles de trabajar ===
    # Por ejemplo: Antes -> Categoria Sexo: [Hombre, Mujer] | Después -> Categoria Sexo: [0, 1].
    df_model = pd.get_dummies(df_model, drop_first=True)

    # === Tenemos que separar las carectreristicas [X], del objetivo [Y] ===

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

    # Pasamos los datosd dfel entrenamiento
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
    print("\n--- Resultados de la Regresión Lineal Múltiple ---")
    print(f"Variables de Entrada (X) usadas: {list(X.columns)}")
    print(f"Variable a Predecir (Y): {target_y}")
    print("-" * 50)
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE (Error): {rmse:.2f}")


    # === Graficacion de prueba ===
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Gráfico de dispersión: Real vs. Predicho
    ax.scatter(Y_test, Y_pred, alpha=0.6, color='darkblue')

    # Línea de ajuste perfecto
    min_val = Y_test.min()
    max_val = Y_test.max()
    ax.plot([min_val, max_val], [min_val, max_val], 
            color='red', linestyle='--', linewidth=2, label='Ajuste Perfecto')

    ax.set_title("Regresión Lineal: Valores Reales vs. Predichos", fontsize=14)
    ax.set_xlabel(f"Valores Reales ({target_y})", fontsize=12)
    ax.set_ylabel(f"Valores Predichos ({target_y})", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)

    # Retornamos la grafica
    return {
        "R2_Score": r2,
        "RMSE": rmse,
        "Coefficients": model.coef_,
        "Plot_Figure": fig
    }

if __name__ == "__main__":
    main()
