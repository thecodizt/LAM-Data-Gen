import streamlit as st
import yaml
import pandas as pd
from pydantic import ValidationError

from config_model import Config, TableType, Distribution

import sys
sys.path.append('../')

from syndatagen.without_sample.cross_sectional.random import RandomGenerator_WithoutSampleCrossSectional

def ui_input():
    params = None
    tables = []

    with_sample = st.sidebar.selectbox("Have Sample Data?", ["Yes", "No"])

    table_type = st.sidebar.selectbox("Type of your Table", ["Cross Sectional", "Time Series"])
    # UI for entering values
    param1 = st.number_input('Enter first parameter', value=1.0)
    param2 = st.number_input('Enter second parameter', value=2.0)
    params = {'param1': param1, 'param2': param2}

    return params, tables

def yaml_input():
    params = None
    tables = []

    # YAML config file
    # uploaded_file = st.file_uploader("Choose a YAML file", type=["yaml", "yml"])
    uploaded_file = uploaded_file = open("../test/test_config.yaml", "r")
    if uploaded_file is not None:
        try:
            params = Config(**yaml.safe_load(uploaded_file)).model_dump()
            with st.expander("Uploaded Parameters", expanded=False):
                st.write(params)

            if params['sample_data']:
                st.subheader("Upload Sample Data")

                for _ in range(params['num_tables']):
                    tables.append(None)

                for i, table in enumerate(params['with_sample_tables']):
                    st.write(table['name'])
                    key = f"file_uploader_{i}"
                    uploaded_file = st.file_uploader(f"Choose a CSV file for {table['name']}", type=["csv"], key=key)
                    if uploaded_file is not None:
                        df = pd.read_csv(uploaded_file)
                        st.write(df)
                        tables[i] = df

        except ValidationError as e:
            st.error(f'Invalid configuration: {e}')

    return params, tables

def without_sample_cross_sectional_generation(param):
    # TODO: Without Sample Cross Sectional Generation
    generated_table = None
    if param['generation_method'] == "random":
        st.write("Method: Random Generation")
        generator = RandomGenerator_WithoutSampleCrossSectional(param)
        generated_table = generator.generate()
    return generated_table

def without_sample_time_series_generation(param):
    # TODO: Without Sample Time Series Generation
    
    return None

# TODO: With sample data generation
def with_sample_cross_sectional_generation(param, table):
    return None

def with_sample_time_series_generation(param, table):
    return None


if __name__ == "__main__":
    st.title("Synthetic Data Generator")

    # Sidebar for input method
    input_method = st.sidebar.selectbox("Choose input method", ["YAML", "UI"])

    params = None
    tables = []

    if input_method == "UI":
        ui_input()

    elif input_method == "YAML":
        params, tables = yaml_input()

    generate_button = True

    # if params is not None:
    #     generate_button = st.button("Generate")

    if params is not None and generate_button:
        if params['sample_data']:
            # TODO: Generation with sample data
            st.subheader("Sample Data")
            for table in tables:
                st.write(table)
        else:
            st.subheader("Generated Data")
            for table in params['without_sample_tables']:
                if table['table_type'] == TableType.CROSS_SECTIONAL:
                    st.subheader(f"{table['name']} - Cross Sectional")
                    generated_table = without_sample_cross_sectional_generation(table)
                    if generated_table is not None:
                        st.write(generated_table)

                elif table['table_type'] == "time_series":
                    st.subheader(f"{table['name']} - Time Series")
                    generated_table = without_sample_time_series_generation(table)
                    if generated_table is not None:
                        st.write(generated_table)
