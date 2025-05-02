# File: NLP_Review.py (Enhanced)

import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
import os
import warnings
from io import BytesIO
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE

# Setup
warnings.filterwarnings("ignore")
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_dir)
nltk.download("punkt", download_dir=nltk_data_dir)
nltk.download("stopwords", download_dir=nltk_data_dir)
nltk.download("wordnet", download_dir=nltk_data_dir)
nltk.download("omw-1.4", download_dir=nltk_data_dir)
_ = PunktSentenceTokenizer()

# -----------------------------------------
# Functions
# -----------------------------------------

def load_data(filepath):
    return pd.read_csv(filepath)

def extract_reviews(df):
    return df['review_body'].dropna().tolist()

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer("english")
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = text.lower().split()
    clean_tokens = [stemmer.stem(w.strip()) for w in tokens if w not in stop_words]
    return clean_tokens

def get_tfidf_matrix(reviews, max_df=0.99, min_df=0.01, max_features=1000):
    tfidf = TfidfVectorizer(
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
        tokenizer=preprocess_text
    )
    tfidf_matrix = tfidf.fit_transform(reviews)
    return tfidf_matrix, tfidf

def generate_wordcloud(text_data):
    all_text = " ".join(text_data)
    return WordCloud(width=800, height=400, background_color='white').generate(all_text)

def perform_lda(matrix, feature_names, num_topics=5, num_words=10):
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(matrix)
    topics = []
    for idx, topic in enumerate(lda.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
        topics.append(f"Topic {idx+1}: " + ", ".join(topic_words))
    return topics

def plot_tsne(matrix, labels=5):
    model = KMeans(n_clusters=labels, random_state=42)
    cluster_labels = model.fit_predict(matrix)
    tsne = TSNE(n_components=2, random_state=42, perplexity=40)
    reduced_data = tsne.fit_transform(matrix.toarray())
    return reduced_data, cluster_labels, model

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')



def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    output.seek(0)
    return output


# -----------------------------------------
# Streamlit App
# -----------------------------------------

st.set_page_config(page_title="NLP & Topic Modeling Dashboard", layout="wide")
st.title(" NLP & Topic Modeling Dashboard")

# Sidebar
st.sidebar.title(" Configuration")
num_topics = st.sidebar.slider("Number of LDA Topics", 2, 10, 5)
max_df = st.sidebar.slider("Max DF (Remove overly common words)", 0.5, 1.0, 0.99)
min_df = st.sidebar.slider("Min DF (Remove rare words)", 0.0, 0.1, 0.01)
max_features = st.sidebar.slider("Max Features", 100, 3000, 1000)

uploaded_file = st.file_uploader(" Upload a CSV file with a 'review_body' column", type=['csv'])

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    st.warning(" Please upload a CSV file to proceed.")
    st.stop()


reviews = extract_reviews(df)
st.subheader(" Sample Reviews")
st.write(reviews[:3])

# WordCloud
st.subheader("WordCloud")
wc = generate_wordcloud(reviews)
fig_wc, ax_wc = plt.subplots()
ax_wc.imshow(wc, interpolation="bilinear")
ax_wc.axis("off")
st.pyplot(fig_wc)

# TF-IDF + LDA
tfidf_matrix, tfidf_model = get_tfidf_matrix(reviews, max_df, min_df, max_features)
st.success(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")

feature_names = tfidf_model.get_feature_names_out()
topics = perform_lda(tfidf_matrix, feature_names, num_topics=num_topics)
st.subheader(" Topics from LDA")
for t in topics:
    st.markdown(f"- {t}")

# Clustering with t-SNE
st.subheader(" t-SNE Clustering")
reduced_data, cluster_labels, model = plot_tsne(tfidf_matrix)
fig_tsne, ax_tsne = plt.subplots()
scatter = ax_tsne.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='tab10')
legend_labels = [f"Cluster {i}" for i in range(model.n_clusters)]
handles = [plt.Line2D([], [], marker='o', color=scatter.cmap(i / model.n_clusters), linestyle='', label=label) for i, label in enumerate(legend_labels)]
ax_tsne.legend(handles=handles, title="Legend", loc='upper right')
st.pyplot(fig_tsne)

# Optional Data Export
st.subheader(" Export Clustered Data")
df_result = pd.DataFrame({
    "review_body": reviews,
    "cluster_label": cluster_labels
})
st.download_button(" Download CSV", convert_df_to_csv(df_result), "clustered_reviews.csv", "text/csv")
st.download_button(" Download Excel", convert_df_to_excel(df_result), "clustered_reviews.xlsx", "application/vnd.ms-excel")

st.success(" Done! All tasks executed successfully.")
