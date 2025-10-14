import streamlit as st
import pandas as pd

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
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")

if __name__ == "__main__":
    main()
