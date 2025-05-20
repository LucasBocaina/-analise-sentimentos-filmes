import streamlit as st
import pandas as pd
import requests
from textblob import TextBlob
from deep_translator import GoogleTranslator
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup
import os
import json
import concurrent.futures
from functools import lru_cache
import time
from requests.exceptions import RequestException
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

CACHE_PATH = "cache/"
if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH)

def buscar_filmes_cache(api_key, total=20):  
    cache_file = f"{CACHE_PATH}filmes.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    filmes = []
    pagina = 1
    while len(filmes) < total:
        url = f"{BASE_URL}/discover/movie?api_key={api_key}&language=pt-BR&page={pagina}"
        resposta = requests.get(url)
        if resposta.status_code != 200:
            break
        dados = resposta.json()
        for filme in dados['results']:
            filmes.append({
                'id': filme['id'],
                'filme': filme['title'],
                'imagem': filme['poster_path'],
                'popularidade': filme['popularity'],
                'nota': filme['vote_average'],
                'url': f"https://www.themoviedb.org/movie/{filme['id']}"
            })
            if len(filmes) >= total:
                break
        pagina += 1
    
    # Salvar o cache para reutilização
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(filmes, f, ensure_ascii=False, indent=2)
    
    return filmes

def buscar_comentarios_async(api_key, movie_id):
    url = f"{BASE_URL}/movie/{movie_id}/reviews?api_key={api_key}&language=en"
    resposta = requests.get(url)
    if resposta.status_code != 200:
        return []
    dados = resposta.json()
    return [review.get("content", "") for review in dados.get("results", []) if review.get("content", "")]

def processar_comentarios_async(filmes, api_key):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        comentarios = list(executor.map(lambda filme: buscar_comentarios_async(api_key, filme['id']), filmes))
    return comentario

def make_request_with_retry(url, retries=3, delay=2):
    for _ in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response
        except RequestException:
            time.sleep(delay)
    return None

# --- CONFIGURAÇÃO API ---
BASE_URL = "https://api.themoviedb.org/3"
IMG_BASE_URL = "https://image.tmdb.org/t/p/w500"
API_KEY = "d6ee0de4eb578a56f6963c90977da780"

# --- CACHE DE FILMES SEM COMENTÁRIOS ---
CACHE_FILMES_SEM_COMENTARIO = "filmes_sem_comentario.json"
CACHE_FILMES_ANALISADOS = "filmes_cache.json"

def carregar_filmes_sem_comentario():
    if os.path.exists(CACHE_FILMES_SEM_COMENTARIO):
        with open(CACHE_FILMES_SEM_COMENTARIO, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def salvar_filmes_sem_comentario(cache):
    with open(CACHE_FILMES_SEM_COMENTARIO, "w", encoding="utf-8") as f:
        json.dump(list(cache), f, ensure_ascii=False, indent=2)

def carregar_cache_filmes_analisados():
    if os.path.exists(CACHE_FILMES_ANALISADOS):
        with open(CACHE_FILMES_ANALISADOS, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def salvar_cache_filmes_analisados(dados):
    with open(CACHE_FILMES_ANALISADOS, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

filmes_sem_comentario = carregar_filmes_sem_comentario()


GIRIAS_EN = {
    "awesome": "great", "sucks": "bad", "boring": "dull", "dope": "cool",
    "lame": "bad", "cringe": "awkward", "meh": "not interesting", "lol": "funny",
    "wtf": "what the hell", "omg": "oh my god", "cool": "nice", "goat": "greatest",
    "fire": "amazing", "trash": "bad", "okish": "okay", "hella": "very", "kinda": "kind of"
}

def corrigir_girias(texto):
    for g, c in GIRIAS_EN.items():
        texto = texto.replace(g, c)
    return texto

@lru_cache(maxsize=10000)
def traduzir_com_cache(texto):
    try:
        return GoogleTranslator(source='auto', target='en').translate(texto)
    except Exception:
        return texto

# Função para análise de sentimentos usando VADER
def classificar_sentimento_vader(texto):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(texto)
    return sentiment_score['compound']  # Retorna a pontuação 'compound'

# Função para análise de sentimentos usando TextBlob
def classificar_sentimento_textblob(texto):
    blob = TextBlob(texto)
    return blob.sentiment.polarity  # Retorna a polaridade de -1 a 1

# Função para análise de sentimentos usando Transformers (DistilBERT)
def classificar_sentimento_transformers(texto):
    sentiment_analyzer = pipeline("sentiment-analysis")
    result = sentiment_analyzer(texto)
    return result[0]['label'], result[0]['score']  # Retorna o label e o score de confiança

# Função para combinar as análises de VADER, TextBlob e Transformers
def classificar_sentimento_combinado(texto):
    # Análise de sentimentos com VADER
    polaridade_vader = classificar_sentimento_vader(texto)
    
    # Análise de sentimentos com TextBlob
    polaridade_textblob = classificar_sentimento_textblob(texto)
    
    # Análise de sentimentos com Transformers
    label_transformers, score_transformers = classificar_sentimento_transformers(texto)
    
    # Normaliza o resultado do Transformers para a escala de -1 a 1
    if label_transformers == 'POSITIVE':
        polaridade_transformers = score_transformers  # A confiança positiva é a polaridade
    else:
        polaridade_transformers = -score_transformers  # Para sentimentos negativos, a polaridade será negativa
    
    # Média das polaridades (pode ser ajustada com pesos se necessário)
    polaridade_media = (polaridade_vader + polaridade_textblob + polaridade_transformers) / 3
    
    # Classificação final baseada na média das polaridades
    if polaridade_media <= -0.15:
        return "😠 Péssimo", polaridade_media
    elif -0.15 < polaridade_media <= 0:
        return "🙁 Ruim", polaridade_media
    elif polaridade_media == 0:
        return "😐 Neutro", polaridade_media
    elif 0 < polaridade_media <= 0.15:
        return "🙂 Bom", polaridade_media
    else:
        return "😃 Ótimo", polaridade_media


def buscar_filmes(api_key, total=10):  
    filmes = []
    pagina = 1
    while len(filmes) < total:
        url = f"{BASE_URL}/discover/movie?api_key={api_key}&language=pt-BR&page={pagina}"
        resposta = requests.get(url)
        if resposta.status_code != 200:
            break
        dados = resposta.json()
        for filme in dados['results']:
            filmes.append({
                'id': filme['id'],
                'filme': filme['title'],
                'imagem': filme['poster_path'],
                'popularidade': filme['popularity'],
                'nota': filme['vote_average'],
                'url': f"https://www.themoviedb.org/movie/{filme['id']}"
            })
            if len(filmes) >= total:
                break
        pagina += 1
    return filmes

def buscar_info_filme(api_key, movie_id):
    url = f"{BASE_URL}/movie/{movie_id}?api_key={api_key}&language=pt-BR"
    resposta = requests.get(url)
    if resposta.status_code != 200:
        return {
            'genero': [], 'data_lancamento': 'N/A',
            'atores': [], 'duracao': 'N/A', 'sinopse': 'Sinopse não disponível'
        }
    dados = resposta.json()
    return {
        'genero': [g['name'] for g in dados.get('genres', [])],
        'data_lancamento': dados.get('release_date', 'N/A'),
        'atores': [],  # Placeholder
        'duracao': dados.get('runtime', 'N/A'),
        'sinopse': dados.get('overview', 'Sinopse não disponível')
    }

def buscar_comentarios(api_key, movie_id):
    url = f"{BASE_URL}/movie/{movie_id}/reviews?api_key={api_key}&language=en"
    resposta = requests.get(url)
    if resposta.status_code != 200:
        return []
    dados = resposta.json()
    return [review.get("content", "") for review in dados.get("results", []) if review.get("content", "")]

def analisar_filmes(api_key):
    filmes = buscar_filmes(api_key)
    cache_existente = carregar_cache_filmes_analisados()
    dados_sentimento = cache_existente.copy()
    filmes_processados_ids = set(entry["url"].split("/")[-1] for entry in cache_existente)
    novos_filmes_sem_comentario = set()

    for filme in filmes:
        if str(filme["id"]) in filmes_processados_ids or str(filme["id"]) in filmes_sem_comentario:
            continue

        comentarios = buscar_comentarios(api_key, filme["id"])
        if not comentarios:
            novos_filmes_sem_comentario.add(str(filme["id"]))
            continue

        info_filme = buscar_info_filme(api_key, filme["id"])

        for idx, texto in enumerate(comentarios):
            try:
                # Usando a função combinada de classificação de sentimentos
                sentimento, polaridade = classificar_sentimento_combinado(texto)

                dados_sentimento.append({
                    "filme": filme["filme"],
                    "comentario": texto,
                    "comentario_original": texto,
                    "polaridade": round(polaridade, 2),
                    "sentimento": sentimento,
                    "imagem": filme['imagem'],
                    "popularidade": filme['popularidade'],
                    "nota": filme['nota'],
                    "genero": ", ".join(info_filme['genero']),
                    "data_lancamento": info_filme['data_lancamento'],
                    "atores": ", ".join(info_filme['atores']),
                    "duracao": info_filme['duracao'],
                    "sinopse": info_filme['sinopse'],
                    "url": filme['url'],
                    "comentario_idx": idx + 1
                })
            except Exception:
                continue

    # Atualiza caches
    salvar_cache_filmes_analisados(dados_sentimento)
    if novos_filmes_sem_comentario:
        filmes_sem_comentario.update(novos_filmes_sem_comentario)
        salvar_filmes_sem_comentario(filmes_sem_comentario)

    return pd.DataFrame(dados_sentimento)

# --- STREAMLIT APP ---
st.set_page_config(page_title="Análise de Sentimentos de Filmes", layout="wide")
st.title("🎬 Análise de Sentimentos de Comentários de Filmes Populares")

with st.spinner("🔍 Carregando e analisando os comentários..."):
    df = analisar_filmes(API_KEY)

if not df.empty:
    filmes_unicos = df['filme'].unique()
    filme_selecionado = st.selectbox("Escolha um filme:", filmes_unicos)

    filme_detalhes = df[df['filme'] == filme_selecionado].iloc[0]
    st.image(IMG_BASE_URL + filme_detalhes['imagem'], width=200)
    st.markdown(f"Popularidade: {filme_detalhes['popularidade']}")
    st.markdown(f"Nota: {filme_detalhes['nota']}")
    st.markdown(f"Gênero: {filme_detalhes['genero']}")
    st.markdown(f"Data de Lançamento: {filme_detalhes['data_lancamento']}")
    st.markdown(f"Duração: {filme_detalhes['duracao']} minutos")
    st.markdown(f"Sinopse: {filme_detalhes['sinopse']}")
    st.markdown(f"URL: [Clique aqui para mais detalhes]({filme_detalhes['url']})")

    st.subheader("📊 Distribuição de Sentimentos dos Comentários")
    grafico_pizza = px.pie(
        df[df['filme'] == filme_selecionado],
        names='sentimento',
        title='Distribuição dos Sentimentos',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(grafico_pizza, use_container_width=True)

    st.subheader("📈 Evolução dos Sentimentos por Filme")
    fig = go.Figure()
    dados_filme = df[df['filme'] == filme_selecionado]
    fig.add_trace(go.Scatter(
        x=dados_filme.index,
        y=dados_filme['polaridade'],
        mode='markers+lines',
        name=filme_selecionado,
        marker=dict(size=10, line=dict(width=2))
    ))
    fig.update_layout(
        title="Polaridade dos Comentários ao Longo do Tempo",
        xaxis_title="Comentários",
        yaxis_title="Polaridade",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Nota Média por Gênero")
    genero_notas = df.groupby("genero")['nota'].mean().reset_index()
    grafico_genero = px.bar(
        genero_notas,
        x='genero',
        y='nota',
        title="Nota Média dos Filmes por Gênero",
        color='nota',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    st.plotly_chart(grafico_genero, use_container_width=True)

    st.markdown("## 💬 Comentários e Análises Detalhadas")
    for i, row in df[df['filme'] == filme_selecionado].sort_values("comentario_idx").iterrows():
        st.markdown(f"Comentário {row['comentario_idx']}: {row['comentario']}")        
        st.write(f"Sentimento: {row['sentimento']} | Polaridade: {row['polaridade']}")

else:
    st.error("❌ Nenhum dado carregado da API.")
