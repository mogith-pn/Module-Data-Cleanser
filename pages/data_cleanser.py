import streamlit
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from clarifai.modules.css import ClarifaiStreamlitCSS
from utils import *



st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)

query_params = st.experimental_get_query_params()
PAT = query_params["pat"][0] if "pat" in query_params.keys() else st.secrets["pat"]
#model = https://clarifai.com/clarifai/main/models/BAAI-bge-base-en
st.markdown("""
<style>
    .stTabs [data-baseweb="tab"] p {
        font-size: 24px;
        font-weight: bold;
    }
    .st-cd {
        gap: 3rem;
    }
</style>""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align: center; color: black;'> Data cleaning & Anomaly detection Assistance using LLM 📂</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "![Clarifai logo](https://www.clarifai.com/hs-fs/hubfs/logo/Clarifai/clarifai-740x150.png?width=180)"
)

uploaded_files = st.file_uploader(
                "_", accept_multiple_files=False, label_visibility="hidden")

with st.sidebar:
    embedding_model = st.text_input("**Enter Embedding model community URL**", key="model_url")
    no_nns = st.number_input("**Enter number of nearest neighbours**", key="no_nns", min_value=3, max_value=25)
    no_of_trees = st.number_input("**Enter number of trees for Annoy tree**", key="no_of_trees", min_value=10, max_value=1000)
    metric_cal =  st.selectbox("**Select metric for Annoy tree,**", ["euclidean", "angular", "manhattan", "dot"], key="metric")
    if "config" not in st.session_state.keys():
      st.session_state["config"] = True

if "compute" not in st.session_state.keys():
    st.session_state["compute"] = False
    
if uploaded_files and st.session_state["config"] :
    df=pd.read_csv(uploaded_files)
    st.write("**Preview of dataframe**")
    st.write(df.head(5))
    pre_compute = df.nunique()
    df['row_content'] = df.apply(lambda row: create_row_content(row, df.columns), axis=1)
    if df.isnull().values.any():
        st.session_state["compute"] = True

    else:
        st.write("No missing values in the data")
        if "embeddings" not in st.session_state.keys():
            with st.spinner("Computing embeddings"):
                embeddings = get_embeddings_for_row_content(embedding_model, PAT, df['row_content'].tolist())
            st.session_state["embeddings"] = embeddings
        marker = st.toggle("Detect and visualize anomaly")

        if marker and st.session_state["embeddings"]:
            with st.form(key='clusters-app'):
                set_flag= False
                umap_n_neighbours=st.slider('**No of neighbours ([See more](https://pair-code.github.io/understanding-umap/)**) :', 2, 100)
                umap_min = st.text_input('**Minimum distance for UMAP [See more](https://pair-code.github.io/understanding-umap/):**', 0.5)
                
                cluster_min_samples = st.slider('**Min Samples (No of samples to be considered as cluster in DBSACN [See more](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html)):**', 3, 100, 5)
                cluster_min_distance = st.text_input('**Epsilon (min distance) [See more](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html):**', 0.5)
                submitted = st.form_submit_button('**Begin clustering!**')
                #ingest_input = run_async_function(insert_inputs_embeddings_to_clarifai, df['row_content'].tolist(), vector_dict, PAT, APP_ID)
                
            if submitted:
                df['embeddings'] = st.session_state["embeddings"]
                with st.spinner("Reducing dimensions with UMAP..."):
                    reduced_dim_list=get_umap_embedding(st.session_state["embeddings"], int(umap_n_neighbours),umap_min)
                    df_reduced_dim=pd.DataFrame(reduced_dim_list,columns=['x','y'])
                with st.spinner('clustering with DBSCAN...'):
                    X = StandardScaler().fit_transform(reduced_dim_list)
                    dbscan = DBSCAN(eps=float(cluster_min_distance), min_samples=cluster_min_samples)
                    df['cluster'] = dbscan.fit_predict(X)
                    df_reduced_dim["cluster"]=df['cluster'].values.tolist()
                    st.scatter_chart(df_reduced_dim, x='x', y='y', color = 'cluster')
                    set_flag = True
            if set_flag:
                download_csv(df)

    if ("embeddings" not in st.session_state.keys()):
        with st.spinner("Computing embeddings"):
            embeddings = get_embeddings_for_row_content(embedding_model, PAT, df['row_content'].tolist())
        st.session_state["embeddings"] = embeddings
    
    if st.session_state["compute"] and st.session_state["embeddings"] :
        with st.spinner("Computing ANNOY index"):
            df['embeddings'] = st.session_state["embeddings"]
            df["nearest_neighbours"] = 0.0
            vector_dict = np.array(st.session_state["embeddings"])
            search_index = build_annoy_tree(vector_dict, metric_cal, no_of_trees)
            df =  find_nearest_neighbour(df, search_index, no_nns)
            df.drop(columns=["row_content","embeddings"], inplace=True)
            post_compute = df.nunique()
            

            # Concatenate the two dataframes
            st.write("##")
            data = pd.concat([pre_compute, post_compute], axis=1, keys=['pre_compute', 'post_compute'])
            st.write("**Processed values before and after imputation**")
            st.write(data)
            # Display the plot in streamlit
            st.bar_chart(data, color=["#FF0000", "#0000FF"] )
            
            st.write("#")
            download_csv(df)


