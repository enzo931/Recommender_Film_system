import streamlit as st
import pandas as pd
from src.data_loader import load_movielens_data
from src.hybrid import HybridRecommender

st.set_page_config(page_title="Recomendador Híbrido de Filmes", layout="wide")

st.title("🎬 Sistema de Recomendação Híbrido de Filmes")
st.markdown("Combine **Conteúdo (TF-IDF)** e **Filtragem Colaborativa (SVD)** para receber sugestões personalizadas do dataset **MovieLens 100k**.")

@st.cache_resource
def init_system():
    df_movies, df_ratings = load_movielens_data()
    return df_movies, df_ratings

with st.spinner("Carregando base de dados do MovieLens..."):
    df_movies, df_ratings = init_system()

# Sidebar de Controles
st.sidebar.header("⚙️ Configurações da Recomendação")

user_list = sorted(df_ratings['user_id'].unique())
movie_list = sorted(df_movies['title'].unique())

selected_user = st.sidebar.selectbox("Selecione o ID do Usuário:", user_list, index=0)
selected_movie = st.sidebar.selectbox("Selecione um Filme que você gosta:", movie_list, index=0)

alpha = st.sidebar.slider(
    "Peso do Conteúdo (Alpha):",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1,
    help="1.0 considera apenas gênero/descrição. 0.0 considera apenas preferências colaborativas dos usuários."
)

top_n = st.sidebar.slider("Quantidade de recomendações:", 3, 10, 5)

if st.sidebar.button("Gerar Recomendações"):
    with st.spinner("Treinando modelos e gerando previsões..."):
        hybrid_system = HybridRecommender(df_movies, df_ratings, alpha=alpha)
        hybrid_system.fit()
        recs = hybrid_system.get_recommendations(selected_user, selected_movie, top_n=top_n)

    st.subheader(f"Top {top_n} recomendações para o Usuário {selected_user} baseadas em '{selected_movie}':")
    
    st.dataframe(
        recs.rename(columns={
            'title': 'Título do Filme',
            'genres': 'Gêneros',
            'hybrid_score': 'Score Híbrido',
            'cb_score': 'Similaridade Conteúdo',
            'cf_pred_rating': 'Nota Prevista (SVD)'
        }),
        use_container_width=True
    )