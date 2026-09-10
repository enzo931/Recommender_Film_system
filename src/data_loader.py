import os
import ssl
import urllib.request
import zipfile
import pandas as pd

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

def download_movielens_data():
    """Baixa e extrai o dataset MovieLens 100k ignorando erros de certificado SSL."""
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DIR, "ml-100k.zip")
    extracted_folder = os.path.join(RAW_DIR, "ml-100k")

    if not os.path.exists(extracted_folder):
        print("Baixando dataset MovieLens 100k...")
        
        # Cria um contexto SSL sem verificação de certificado expirado
        context = ssl._create_unverified_context()
        
        with urllib.request.urlopen(MOVIELENS_URL, context=context) as response, open(zip_path, 'wb') as out_file:
            out_file.write(response.read())
        
        print("Extraindo arquivos...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
        
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
        print("Download e extração concluídos!")

def load_movielens_data():
    """Carrega e trata os DataFrames de filmes e avaliações do MovieLens 100k."""
    download_movielens_data()
    
    extracted_folder = os.path.join(RAW_DIR, "ml-100k")
    
    # Carrega catálogo de filmes (u.item)
    movies_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
                   'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                   'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                   'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    
    df_movies = pd.read_csv(
        os.path.join(extracted_folder, "u.item"), 
        sep="|", 
        encoding="latin-1", 
        names=movies_cols
    )
    
    # Mapeia colunas de gêneros em formato texto para o TF-IDF
    genre_cols = movies_cols[5:]
    def extract_genres(row):
        return " ".join([genre for genre in genre_cols if row[genre] == 1])
    
    df_movies['genres'] = df_movies.apply(extract_genres, axis=1)
    df_movies = df_movies[['item_id', 'title', 'genres']]

    # Carrega registro de avaliações (u.data)
    ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    df_ratings = pd.read_csv(
        os.path.join(extracted_folder, "u.data"), 
        sep="\t", 
        names=ratings_cols
    )
    df_ratings = df_ratings[['user_id', 'item_id', 'rating']]

    return df_movies, df_ratings