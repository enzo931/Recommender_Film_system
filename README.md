# 🎬 Hybrid Movie Recommender System

Um sistema de recomendação híbrido de filmes construído em Python, utilizando o dataset público **MovieLens 100k**. O projeto combina duas abordagens clássicas da indústria de tecnologia: **Filtragem Baseada em Conteúdo** (TF-IDF e Similaridade por Cosseno) e **Filtragem Colaborativa** (Fatoração de Matrizes via SVD).

Possui execução via linha de comando (CLI) e uma **interface web interativa** desenvolvida com Streamlit.

---

## 📌 Como Funciona

O modelo calcula uma pontuação híbrida ponderada para combinar as duas inteligências:

$$\text{Score Híbrido} = (\alpha \times \text{Score Conteúdo}) + ((1 - \alpha) \times \text{Nota Prevista SVD})$$

1. **Content-Based Filtering:** Vetoriza as palavras-chave dos gêneros dos filmes com `TfidfVectorizer` e calcula a similaridade cosseno entre o título selecionado e os demais itens do catálogo.
2. **Collaborative Filtering:** Treina um algoritmo de Decomposição em Valores Singulares (`SVD`) sobre 100.000 avaliações de usuários para prever a nota de itens ainda não vistos.
3. **Engine Híbrida:** Normaliza ambas as métricas na mesma escala (0 a 1) e aplica o peso $\alpha$ configurável para gerar a lista final de recomendações.

---

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python 3.10+
- **Manipulação de Dados:** Pandas, NumPy
- **Machine Learning & NLP:** Scikit-learn, Scikit-surprise, SciPy
- **Interface Web:** Streamlit
- **Dataset:** MovieLens 100k (1.682 filmes e 100.000 avaliações)

---

## 📂 Estrutura do Repositório

```text
recommender-system/
├── data/              # Armazenamento local do dataset
│   └── raw/           # Dataset MovieLens 100k extraído
├── src/               # Código-fonte e algoritmos
│   ├── data_loader.py # Download, extração e tratamento dos dados
│   ├── content_based.py # Lógica do recomendador por conteúdo
│   ├── collaborative.py # Lógica do recomendador colaborativo (SVD)
│   └── hybrid.py      # Recomendador híbrido ponderado
├── app.py             # Interface gráfica interativa (Streamlit)
├── main.py            # Ponto de entrada CLI
├── requirements.txt   # Dependências do projeto
└── README.md
```

#  🚀 Como Executar o Projeto Localmente

## 1. Clonar o Repositório
```
git clone [https://github.com/seu-usuario/recommender-system.git](https://github.com/seu-usuario/recommender-system.git)
cd recommender-system
```
## 2. Criar e Ativar o Ambiente Virtual
```
python -m venv venv
```
## Windows (PowerShell):
```
.\venv\Scripts\activate
```
## Linux / macOS:
```
source venv/bin/activate
```
## 3. Instalar as Dependências
```
pip install -r requirements.txt
```
## 4. Executar via Terminal (CLI)
Para testar a pipeline de Machine Learning diretamente no terminal:
```
python main.py
```
## 5. Executar a Interface Web (Streamlit)
Para abrir a aplicação interativa no seu navegador:
```
streamlit run app.py
```
