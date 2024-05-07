import os
import streamlit as st
import pandas as pd
from clarifai.client.model import Model
from clarifai.client.input import Inputs

def create_row_content(row, df_columns):
    row_content = []
    for col in df_columns :
        row_content.append(f"{col.replace(' ', '_')} is {row[col]}")
    return ', '.join(row_content)


def get_embeddings_for_row_content(model_name: str, _PAT, row_content):
    try:
        model_obj = Model(model_name, pat =_PAT)
        input_obj = Inputs(pat = _PAT)
        batch_size = 32
        embeddings = []
        print(f" len of row content {len(row_content)}")

        for i in range(0, len(row_content), batch_size):
            batch = row_content[i : i + batch_size]
            print(f"computing embeddings {i} to {i + batch_size}")  # change this  to nice display in UI
            input_batch = [
                input_obj.get_text_input(input_id=str(id), raw_text=inp)
                for id, inp in enumerate(batch)
            ]
            predict_response = model_obj.predict(input_batch)
            embeddings.extend(
                [
                    list(output.data.embeddings[0].vector)
                    for output in predict_response.outputs
                ]
            )
        return embeddings

    
    except Exception as e:
        st.error(f"Embedding predictions failed due to {e}")
        return None
    
