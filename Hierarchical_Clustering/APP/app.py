import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

import scipy.cluster.hierarchy as sch

# -----------------------------
# LOAD CSS
# -----------------------------
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# -----------------------------
# SAFE DATA LOADING (UPDATED)
# -----------------------------
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except:
        df = pd.read_csv(uploaded_file, encoding="latin1")
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "Financial-data.csv")

    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding="latin1")
    else:
        st.error("Financial-data.csv not found in APP folder.")
        st.stop()

# -----------------------------
# APP TITLE
# -----------------------------
st.markdown("<h1 class='title'>🟣 News Topic Discovery Dashboard</h1>", unsafe_allow_html=True)

st.write(
    "This system uses Hierarchical Clustering to automatically group "
    "similar news articles based on textual similarity."
)

# -----------------------------
# DETECT TEXT COLUMN
# -----------------------------
text_columns = df.select_dtypes(include="object").columns.tolist()

text_col = st.sidebar.selectbox(
    "Select Text Column",
    text_columns
)

# -----------------------------
# TEXT VECTORIZATION SETTINGS
# -----------------------------
st.sidebar.subheader("📝 Text Vectorization")

max_features = st.sidebar.slider(
    "Maximum TF-IDF Features", 100, 2000, 1000
)

use_stopwords = st.sidebar.checkbox("Use English Stopwords", value=True)

ngram_option = st.sidebar.selectbox(
    "N-gram Range",
    ["Unigrams", "Bigrams", "Unigrams + Bigrams"]
)

if ngram_option == "Unigrams":
    ngram_range = (1,1)
elif ngram_option == "Bigrams":
    ngram_range = (2,2)
else:
    ngram_range = (1,2)

# -----------------------------
# HIERARCHICAL SETTINGS
# -----------------------------
st.sidebar.subheader("🌳 Hierarchical Clustering")

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ["ward", "complete", "average", "single"]
)

distance_metric = "euclidean"

dendro_samples = st.sidebar.slider(
    "Number of Articles for Dendrogram",
    20, 200, 100
)

n_clusters = st.sidebar.slider(
    "Number of Clusters",
    2, 10, 3
)

# -----------------------------
# VECTORIZE TEXT
# -----------------------------
stop_words = "english" if use_stopwords else None

vectorizer = TfidfVectorizer(
    max_features=max_features,
    stop_words=stop_words,
    ngram_range=ngram_range,
    min_df=2
)

X = vectorizer.fit_transform(df[text_col].astype(str))
X_dense = X.toarray()

# PCA for visualization
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_dense)

# -----------------------------
# GENERATE DENDROGRAM
# -----------------------------
if st.button("🟦 Generate Dendrogram"):

    st.subheader("🌲 Dendrogram")

    Z = sch.linkage(X_dense[:dendro_samples], method=linkage_method)

    fig = plt.figure(figsize=(10,5))
    sch.dendrogram(Z)
    plt.xlabel("Article Index")
    plt.ylabel("Distance")

    st.pyplot(fig)

# -----------------------------
# APPLY CLUSTERING
# -----------------------------
if st.button("🟩 Apply Clustering"):

    hc = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=distance_metric,
        linkage=linkage_method
    )

    clusters = hc.fit_predict(X_dense)
    df["cluster"] = clusters

    # -----------------------------
    # PCA VISUALIZATION
    # -----------------------------
    st.subheader("📊 Cluster Visualization")

    fig2 = plt.figure(figsize=(7,5))
    plt.scatter(X_2d[:,0], X_2d[:,1], c=clusters)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    st.pyplot(fig2)

    # -----------------------------
    # SILHOUETTE SCORE
    # -----------------------------
    st.subheader("📈 Validation")

    score = silhouette_score(X_dense, clusters)
    st.write("Silhouette Score:", round(score,3))

    if score > 0.5:
        st.success("Clusters are well separated.")
    elif score > 0:
        st.info("Clusters have moderate overlap.")
    else:
        st.error("Poor clustering structure.")

    # -----------------------------
    # CLUSTER SUMMARY
    # -----------------------------
    st.subheader("📑 Cluster Summary")

    terms = vectorizer.get_feature_names_out()
    summary = []

    for c in sorted(df["cluster"].unique()):

        idx = np.where(clusters==c)[0]

        tfidf_mean = X_dense[idx].mean(axis=0)
        top_ids = np.argsort(tfidf_mean)[-10:]
        keywords = [terms[i] for i in top_ids]

        snippet = df.iloc[idx[0]][text_col][:120]

        summary.append({
            "Cluster ID": c,
            "Number of Articles": len(idx),
            "Top Keywords": ", ".join(keywords),
            "Sample Article": snippet
        })

    st.dataframe(pd.DataFrame(summary))

    # -----------------------------
    # BUSINESS INTERPRETATION
    # -----------------------------
    st.subheader("🧠 Business Interpretation")

    for row in summary:
        st.write(
            f"🟣 Cluster {row['Cluster ID']}: Articles share themes around "
            f"{row['Top Keywords'].split(',')[0]} and related topics."
        )
