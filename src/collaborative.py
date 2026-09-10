from surprise import Dataset, Reader, SVD
import pandas as pd

class CollaborativeRecommender:
    def __init__(self, df_ratings):
        self.df_ratings = df_ratings
        self.model = SVD()
        
    def fit(self):
        # Define a escala das avaliações (ex: 1.0 a 5.0)
        reader = Reader(rating_scale=(1.0, 5.0))
        data = Dataset.load_from_df(self.df_ratings[['user_id', 'item_id', 'rating']], reader)
        trainset = data.build_full_trainset()
        
        self.model.fit(trainset)
        
    def predict_rating(self, user_id, item_id):
        prediction = self.model.predict(user_id, item_id)
        return round(prediction.est, 2)