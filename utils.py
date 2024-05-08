import os
import uuid
import streamlit as st
import pandas as pd
import httpx
import asyncio
import umap.umap_ as umap
from annoy import AnnoyIndex
import numpy as np
from clarifai.client.model import Model
from clarifai.client.input import Inputs
from google.protobuf.struct_pb2 import Struct


def create_row_content(row, df_columns):
    row_content = []
    for col in df_columns :
        row_content.append(f"{col.replace(' ', '_')} is {row[col]}")
    return ', '.join(row_content)

def convert_embeddings_to_dict(embeddings):
    return [{f"vector": emb} for i, emb in enumerate(embeddings)]

@st.cache_data
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
    

def build_annoy_tree(embeddings_array, metric, no_of_trees):
    
    annoy_index = AnnoyIndex(embeddings_array.shape[1], metric)
    for i, vector in enumerate(embeddings_array):
        annoy_index.add_item(i, vector)
    annoy_index.build(no_of_trees)
    return annoy_index

def find_nearest_neighbour(dtf, annoy_index, no_of_neighbors):
    print("before isnull")
    if dtf.isnull().sum().any():
        null_value_positions = np.where(dtf.isnull())
        
        for i,j in zip(null_value_positions[0],null_value_positions[1]):
            query_vector = np.array(dtf.loc[i, 'embeddings'])
            nearest_neighbors = annoy_index.get_nns_by_vector(query_vector, no_of_neighbors)
            dtf.loc[i,"nearest_neighbours"]=str(nearest_neighbors)
            neighbour_values=[]
            for nns in nearest_neighbors[1:]:
                neighbour_values.append(dtf.iloc[nns,j])
                if isinstance(neighbour_values[0],str):
                    impute_val = neighbour_values[0]
                else:
                    print(f"impute_val before compute")
                    impute_val = np.nanmean(neighbour_values)
                    print(impute_val)
            
            #print(neighbour_values,impute_val)
            dtf.iloc[i,j]=impute_val
            print("completed this block as well")
            
    return dtf

def run_async_function(func, *args, **kwargs):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(func(*args, **kwargs))
    finally:
        loop.close()

async def insert_inputs_embeddings_to_clarifai(contents, vector_dict, PAT, app_id):
    input_obj = Inputs(pat = PAT, app_id=app_id)
    batch_size = 10
    for idx in range(0, len(contents), batch_size):
        try:
            batch_texts = contents[idx : idx + batch_size]
            batch_metadatas = vector_dict[idx : idx + batch_size] 
            batch_ids = [uuid.uuid4().hex for _ in range(len(batch_texts))]

            if batch_metadatas is not None:
                meta_list = []
                for meta in batch_metadatas:
                    meta_struct = Struct()
                    meta_struct.update(meta)
                    meta_list.append(meta_struct)
            input_batch = [
                input_obj.get_text_input(   
                    input_id=batch_ids[i],
                    raw_text=text,
                    metadata=meta_list[i],
                )
                for i, text in enumerate(batch_texts)
            ]
            result_id = input_obj.upload_inputs(inputs=input_batch)
            
        except Exception as e:
            print(f"Failed to insert inputs due to {e}")
            

def get_umap_embedding(embedding_list , n_neighbors, min_dist ) :
    """Reduces the dimensionality of the embeddings into 3D using UMAP.

    Args:
      embedding_list: A list of embeddings.
      n_neighbors: The number of neighbors to consider.
      
    Returns:
      A ndarray of 3D embeddings.
    """
    reducer = umap.UMAP(random_state=42, n_neighbors=n_neighbors, n_components=2, min_dist=float(min_dist))
    reducer.fit(embedding_list)
    embedding = reducer.embedding_
    return embedding

def download_csv(df):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
            "Download Cleaned file",
            csv,
            "Cleaned_file.csv",
            "text/csv",
            key='download-csv'
            )
