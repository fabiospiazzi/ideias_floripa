# app.py

import streamlit as st
import pandas as pd
import re
import folium
from streamlit_folium import st_folium
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

# Função para carregar CSV do GitHub ou upload
@st.cache_data
def carregar_dados(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        url = "https://raw.githubusercontent.com/fabiospiazzi/ideias_floripa/main/dados_ideias_floripa.csv"
        return pd.read_csv(url)

# Lista fixa de bairros
bairros_floripa = [
    "Centro", "Trindade", "Ingleses do Rio Vermelho", "Canasvieiras", "Rio Tavares",
    "Estreito", "Agronômica", "Capoeiras", "Itacorubi", "Campeche", "Córrego Grande",
    "Jurerê", "Costeira do Pirajubaé", "Lagoa da Conceição", "Saco dos Limões", "Pantanal",
    "Santa Mônica", "João Paulo", "Abraão", "Carianos", "Monte Verde", "Coqueiros",
    "Barra da Lagoa", "Tapera", "Ribeirão da Ilha", "Sambaqui", "Armação", "Ratones"
]

def extrair_bairro(texto):
    texto_lower = texto.lower()
    for bairro in bairros_floripa:
        if re.search(rf"\b{re.escape(bairro.lower())}\b", texto_lower):
            return bairro
    return None

# Geolocalizador
geolocator = Nominatim(user_agent="floripa-sentimento-streamlit")

@st.cache_data
def geocodificar_bairro(bairro):
    try:
        local = geolocator.geocode(f"{bairro}, Florianópolis, SC, Brasil")
        if local:
            return (local.latitude, local.longitude)
    except GeocoderTimedOut:
        time.sleep(1)
        return geocodificar_bairro(bairro)
    return (None, None)

# Carrega modelo e tokenizer
@st.cache_resource
def carregar_modelo():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer), tokenizer

# Função para adicionar nova ideia
def adicionar_ideia(df, nova_ideia, sentiment_analyzer, tokenizer):
    # Analisa o sentimento da nova ideia
    resultado = analisar_sentimento_completo(nova_ideia, sentiment_analyzer, tokenizer)
    
    # Cria um novo DataFrame com a nova ideia
    novo_df = pd.DataFrame({
        'IDEIA': [nova_ideia],
        'sentimento': [resultado[0]],
        'confianca': [resultado[1]],
        'num_tokens': [resultado[2]]
    })
    
    # Extrai bairro e coordenadas
    novo_df['bairro'] = novo_df['IDEIA'].astype(str).apply(extrair_bairro)
    novo_df[['latitude', 'longitude']] = novo_df['bairro'].apply(
        lambda x: pd.Series(geocodificar_bairro(x)) if pd.notna(x) else pd.Series([None, None])
    )
    
    # Concatena com o DataFrame original
    return pd.concat([df, novo_df], ignore_index=True)

def analisar_sentimento_completo(texto, sentiment_analyzer, tokenizer):
    try:
        # Limita o texto a 512 tokens (limitação do BERT)
        resultado = sentiment_analyzer(texto[:512])[0]
        label = resultado.get('label', '')
        score = resultado.get('score', 0.0)
        
        # Converte rótulo de estrelas para sentimento
        match = re.match(r"(\d)\s*stars?", label, re.IGNORECASE)
        if match:
            estrelas = int(match.group(1))
            sentimento = (
                "Negativo" if estrelas <= 2 else
                "Neutro" if estrelas == 3 else
                "Positivo"
            )
            # Conta tokens
            tokens = tokenizer(texto, truncation=True, max_length=512, return_tensors="pt")
            num_tokens = tokens['input_ids'].shape[1]
            return (sentimento, float(score), num_tokens)
        else:
            st.warning(f"Formato de label inesperado: {label}")
            return ("Indefinido", 0.0, 0)
    
    except Exception as e:
        st.error(f"Erro ao analisar sentimento: {str(e)}")
        return ("Erro", 0.0, 0)

# App principal
def main():
    st.set_page_config(page_title="Ideias para Florianópolis", layout="wide")
    st.title("💬 Mapa de Ideias e Sentimentos sobre Florianópolis")

    uploaded_file = st.file_uploader("📁 Faça upload de um CSV com a coluna 'IDEIA'", type=["csv"])
    df = carregar_dados(uploaded_file)

    if 'IDEIA' not in df.columns:
        st.error("O arquivo deve conter a coluna 'IDEIA'")
        return

    # Carrega modelo e tokenizer juntos
    sentiment_analyzer, tokenizer = carregar_modelo()

    # Seção para adicionar nova ideia
    with st.expander("➕ Adicionar Nova Ideia", expanded=True):
        with st.form("nova_ideia_form"):
            nova_ideia = st.text_area("Digite sua ideia sobre Florianópolis:", height=150)
            enviar = st.form_submit_button("Analisar Sentimento")
            
            if enviar and nova_ideia:
                with st.spinner("Analisando nova ideia..."):
                    df = adicionar_ideia(df, nova_ideia, sentiment_analyzer, tokenizer)
                    st.success("Ideia adicionada com sucesso!")
                    st.balloons()

    st.info("Analisando sentimentos... Aguarde.")
    
    # Aplica a análise de sentimento para todas as ideias
    if 'sentimento' not in df.columns:
        df[['sentimento', 'confianca', 'num_tokens']] = df['IDEIA'].astype(str).apply(
            lambda x: pd.Series(analisar_sentimento_completo(x, sentiment_analyzer, tokenizer))
        
        # Extrai bairros e coordenadas
        df['bairro'] = df['IDEIA'].astype(str).apply(extrair_bairro)
        df[['latitude', 'longitude']] = df['bairro'].apply(
            lambda x: pd.Series(geocodificar_bairro(x)) if pd.notna(x) else pd.Series([None, None])
        )

    # Mapa
    mapa = folium.Map(location=[-27.5954, -48.5480], zoom_start=12)
    for _, row in df.dropna(subset=['latitude', 'longitude']).iterrows():
        cor = (
            "green" if row['sentimento'] == "Positivo" else
            "blue" if row['sentimento'] == "Neutro" else
            "red"
        )
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"<b>Bairro:</b> {row['bairro']}<br><b>Sentimento:</b> {row['sentimento']}<br><b>Confiança:</b> {row['confianca']:.2f}",
            icon=folium.Icon(color=cor)
        ).add_to(mapa)

    st.subheader("🗺️ Mapa com Ideias Geolocalizadas")
    st_folium(mapa, width=1000, height=600)

    st.subheader("📊 Tabela de Sentimentos")
    st.dataframe(df[['IDEIA', 'sentimento', 'confianca', 'num_tokens', 'bairro']])

if __name__ == '__main__':
    main()

