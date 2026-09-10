import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender

class HybridRecommender:
    def __init__(self, df_movies, df_ratings, alpha=0.5):
        """
        alpha: Peso da Filtragem Baseada em Conteúdo (0.0 a 1.0).
               1.0 considera apenas conteúdo; 0.0 considera apenas colaborativa.
        """
        self.df_movies = df_movies
        self.df_ratings = df_ratings
        self.alpha = alpha
        
        self.cb = ContentBasedRecommender(df_movies)
        self.cf = CollaborativeRecommender(df_ratings)
        self.scaler = MinMaxScaler()

    def fit(self):
        print("Treinando modelo de Conteúdo...")
        self.cb.fit()
        print("Treinando modelo Colaborativo (SVD)...")
        self.cf.fit()

    def get_recommendations(self, user_id, title, top_n=5):
        if title not in self.df_movies['title'].values:
            return f"Filme '{title}' não encontrado na base de dados."

        # 1. Obtém similaridade de conteúdo para todos os filmes em relação ao título
        idx = self.df_movies[self.df_movies['title'] == title].index[0]
        content_scores = list(enumerate(self.cb.similarity_matrix[idx]))
        
        # Cria DataFrame temporário de candidatos
        candidates = []
        for i, cb_score in content_scores:
            item_id = self.df_movies.iloc[i]['item_id']
            item_title = self.df_movies.iloc[i]['title']
            
            # Descarta o próprio filme de referência
            if item_title == title:
                continue
                
            # Predict de nota do usuário via SVD
            cf_pred_rating = self.cf.predict_rating(user_id, item_id)
            
            candidates.append({
                'item_id': item_id,
                'title': item_title,
                'genres': self.df_movies.iloc[i]['genres'],
                'cb_score': cb_score,
                'cf_pred_rating': cf_pred_rating
            })
            
        df_cand = pd.DataFrame(candidates)
        
        # 2. Normaliza ambas as métricas para a escala [0, 1]
        df_cand[['cb_norm', 'cf_norm']] = self.scaler.fit_transform(
            df_cand[['cb_score', 'cf_pred_rating']]
        )
        
        # 3. Calcula o Score Híbrido Ponderado
        df_cand['hybrid_score'] = (
            self.alpha * df_cand['cb_norm'] + (1 - self.alpha) * df_cand['cf_norm']
        )
        
        # Ordena pelos melhores resultados híbridos
        top_recommendations = (
            df_cand.sort_values(by='hybrid_score', ascending=False)
                   .head(top_n)[['title', 'genres', 'hybrid_score', 'cb_score', 'cf_pred_rating']]
        )
        
        return top_recommendations