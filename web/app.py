import streamlit as st
import yaml
import pandas as pd
from pydantic import ValidationError
import base64

from config_model import Config, TableType, Distribution

import sys
sys.path.append('../')

def ui_input():
    params = {}
    tables = []

    num_tables = st.selectbox('Select number of tables', [1, 2])
    params['num_tables'] = num_tables
    params['sample_data'] = st.checkbox('Sample data', value=True)

    if num_tables == 2:
        params['foreign_keys'] = st.checkbox('Foreign keys', value=True)
        params['foreign_key_col'] = st.text_input('Enter foreign key column', value='PassengerId')
    else:
        params['foreign_keys'] = False

    params['with_sample_tables'] = []
    for i in range(num_tables):
        with st.expander(f'Table {i+1}', expanded=True):
            name = st.text_input(f'Enter name of table {i+1}', value=f'DF{i+1}', key=f'table_name_{i}')
            num_rows = st.number_input(f'Enter number of rows for table {i+1}', min_value=1, value=200, key=f'table_rows_{i}')
            params['with_sample_tables'].append({'name': name, 'num_rows': num_rows})

    if params['sample_data']:
        st.subheader("Upload Sample Data")

        for _ in range(num_tables):
            tables.append(None)

        for i, table in enumerate(params['with_sample_tables']):
            st.write(table['name'])
            key = f"file_uploader_{i}"
            uploaded_file = st.file_uploader(f"Choose a CSV file for {table['name']}", type=["csv"], key=key)
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.write(df)
                tables[i] = df

    return params, tables

def yaml_input():
    params = None
    tables = []

    # YAML config file
    uploaded_file = st.file_uploader("Choose a YAML file", type=["yaml", "yml"])
    # uploaded_file = uploaded_file = open("../test/test_config.yaml", "r")
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

def merge_tables(params, tables):
    df1 = tables[0]
    df2 = tables[1]

    df = pd.merge(df1, df2, on=str(params['foreign_key_col']))

    return df, df1.columns.values.tolist(), df2.columns.values.tolist()

def without_sample_generate_handle(params, table):

    result = table

    return result

def split_tables(params, table, df1_cols, df2_cols):
    foreign_key = params['foreign_key_col']
    df = table

    df1 = df[df1_cols]
    df2 = df[df2_cols]

    # Set the foreign key as the index for each new DataFrame
    df1.set_index(foreign_key, inplace=True)
    df2.set_index(foreign_key, inplace=True)

    return [df1, df2]

    

if __name__ == "__main__":
    st.title("Synthetic Data Generator")

    # Sidebar for input method
    input_method = st.sidebar.selectbox("Choose input method", ["YAML", "UI"])

    params = None
    tables = []

    if input_method == "UI":
        params, tables = ui_input()

    elif input_method == "YAML":
        params, tables = yaml_input()

    generate_button = None

    if params is not None:
        if params['num_tables'] > 2 or params['num_tables'] < 1:
            st.error("Number of tables should be 1 or 2")
        else:
            generate_button = st.button("Generate")

    if params is not None and generate_button:
        if params['sample_data']:

            # TODO: Generation with sample data

            if params['foreign_keys']:
                table, df1_cols, df2_cols = merge_tables(params, tables)

                st.subheader("Merged Data based on Foreign Key")

                st.write(table)

                st.header("Generated Data")

                table = without_sample_generate_handle(params, table)

                tables = split_tables(params, table, df1_cols, df2_cols)

                # Assuming 'tables' is a list of pandas DataFrames
                for i, table in enumerate(tables):
                    st.subheader(f"Table {i+1}")
                    st.write(table)
                    st.download_button(label="Download CSV", data=table.to_csv().encode("utf-8"), file_name=f"table_{i+1}.csv", mime="text/csv")

            else:
                st.header("Generated Data")

                table = without_sample_generate_handle(params, tables[0])
                st.write(table)
                st.download_button(label="Download CSV", data=table.to_csv().encode("utf-8"), file_name=f"generated.csv", mime="text/csv")
        else:
            # TODO: Generation without sample data
            pass