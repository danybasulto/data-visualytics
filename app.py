import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.set_page_config(page_title="Data Visualytics",
                       page_icon="./assets/icon_light.png",
                       layout="wide")

    left_col, center_col, right_col = st.columns(3)

    with center_col:
        st.image("./assets/logo_dark.png", use_container_width=True)

    st.header("Cargar Archivo")
    file = st.file_uploader("Elige un archivo CSV o XLSX:", type=["csv", "xlsx"])

    if file is not None:
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            elif file.name.endswith(".xlsx"):
                df = pd.read_excel(file)

            st.success("Archivo cargado exitosamente!")
            df_display = df.head(100)
            st.dataframe(df_display)
        except Exception as ex:
            st.error(f"Error al cargar el archivo: {ex}")

if __name__ == "__main__":
    main()
