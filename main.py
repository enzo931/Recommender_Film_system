from src.data_loader import load_movielens_data
from src.hybrid import HybridRecommender

def main():
    print("=== Inicializando Sistema de Recomendação Híbrido ===")
    df_movies, df_ratings = load_movielens_data()
    
    print(f"\nBase carregada com sucesso:")
    print(f"- Total de Filmes: {len(df_movies)}")
    print(f"- Total de Avaliações: {len(df_ratings)}")
    
    # Inicializa modelo com peso 0.5 para cada abordagem
    hybrid_system = HybridRecommender(df_movies, df_ratings, alpha=0.5)
    hybrid_system.fit()
    
    target_user_id = 196
    target_movie = "Star Wars (1977)"
    
    print(f"\nGerando top 5 recomendações para o Usuário {target_user_id} com base em '{target_movie}':")
    recommendations = hybrid_system.get_recommendations(target_user_id, target_movie, top_n=5)
    
    print("\n" + recommendations.to_string(index=False))

if __name__ == "__main__":
    main()