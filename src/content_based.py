import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, df_movies):
        self.df_movies = df_movies
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.similarity_matrix = None
        
    def fit(self):
        # Matriz de características baseada nos gêneros/descrição
        tfidf_matrix = self.vectorizer.fit_transform(self.df_movies['genres'])
        # Cálculo de similaridade entre todos os pares de filmes
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
    def get_recommendations(self, title, top_n=3):
        if title not in self.df_movies['title'].values:
            return f"Filme '{title}' não encontrado na base."
            
        idx = self.df_movies[self.df_movies['title'] == title].index[0]
        scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Ordena os filmes por similaridade (exclui o próprio filme)
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        
        recommendations = []
        for i, score in sorted_scores:
            recommendations.append({
                'title': self.df_movies.iloc[i]['title'],
                'similarity_score': round(score, 4)
            })
            
        return pd.DataFrame(recommendations)