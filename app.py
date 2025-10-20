import streamlit as st
import pandas as pd

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

                variable_y = st.selectbox(
                    "Seleccione la variable de la calse (y):",
                    options=options_y,
                    help="Esta es la variable que el modelo intentará predecir."
                )
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")

if __name__ == "__main__":
    main()
