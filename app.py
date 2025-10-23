import streamlit as st
import pandas as pd
import numpy as np

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
                apply_linear_regression(df_display, variables_x, variable_y)

        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")

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
        None: No retorna ningun valor.
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
    print("\n--- Resultados de la Regresión Lineal Múltiple ---")
    print(f"Variables de Entrada (X) usadas: {list(X.columns)}")
    print(f"Variable a Predecir (Y): {target_y}")
    print("-" * 50)
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE (Error): {rmse:.2f}")

    return None

    
if __name__ == "__main__":
    main()
