from surprise.model_selection import cross_validate
from surprise import SVD, Dataset, Reader
from src.data_loader import load_movielens_data

def evaluate_collaborative_model():
    _, df_ratings = load_movielens_data()
    
    reader = Reader(rating_scale=(1.0, 5.0))
    data = Dataset.load_from_df(df_ratings[['user_id', 'item_id', 'rating']], reader)
    
    algo = SVD()
    
    print("Iniciando Validação Cruzada (5-Fold CV)...")
    results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    
    mean_rmse = results['test_rmse'].mean()
    mean_mae = results['test_mae'].mean()
    
    print(f"\nResultado da Avaliação:")
    print(f"- RMSE Médio: {mean_rmse:.4f}")
    print(f"- MAE Médio:  {mean_mae:.4f}")

if __name__ == "__main__":
    evaluate_collaborative_model()