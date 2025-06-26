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

# Adicione no início do seu código (após os imports)
if 'dados_processados' not in st.session_state:
    st.session_state.dados_processados = None

# Função para carregar CSV do GitHub ou upload
@st.cache_data
def carregar_dados(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        st.info("Nenhum arquivo CSV carregado ainda.")
        #url = "https://raw.githubusercontent.com/fabiospiazzi/ideias_floripa/main/dados_ideias_floripa.csv"
        #return pd.read_csv(url)

# Lista fixa de bairros
bairros_floripa = [
    "Abraão", "Agronômica", "Alto Ribeirão", "Armação", "Balneário", "Barra da Lagoa", 
    "Cachoeira do Bom Jesus", "Cacupé", "Campeche", "Canasvieiras", "Canto dos Araçás",
    "Capoeiras", "Carianos", "Carvoeira", "Centro", "Chácara do Espanha", "Coqueiros",
    "Córrego Grande", "Costa de Dentro", "Costeira do Pirajubaé", "Estreito", "Fazenda do Rio Tavares",
    "Ingleses do Rio Vermelho", "Ingleses", "Itacorubi", "Jardim Atlântico", "João Paulo", "Jurerê",
    "Jurerê Internacional", "Lagoa da Conceição", "Monte Cristo", "Monte Verde", "Morro das Pedras",
    "Pantanal", "Pântano do Sul", "Parque São Jorge", "Picadas do Sul", "Picadas", "Ponta das Canas",
    "Praia Brava", "Ratones", "Ribeirão da Ilha", "Rio Tavares", "Rio Vermelho", "Sacramento",
    "Saco dos Limões", "Sambaqui", "Santa Mônica", "Santo Antônio de Lisboa", "São João do Rio Vermelho",
    "Tapera", "Trindade", "Vargem do Bom Jesus", "Vargem Grande", "Vargem Pequena"
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

# Função para analisar sentimento
def analisar_sentimento_completo(texto, sentiment_analyzer, tokenizer):
    try:
        resultado = sentiment_analyzer(texto[:512])[0]
        label = resultado.get('label', '')
        score = resultado.get('score', 0.0)
        
        match = re.match(r"(\d)\s*stars?", label, re.IGNORECASE)
        if match:
            estrelas = int(match.group(1))
            sentimento = (
                "Negativo" if estrelas <= 2 else
                "Neutro" if estrelas == 3 else
                "Positivo"
            )
            tokens = tokenizer(texto, truncation=True, max_length=512, return_tensors="pt")
            num_tokens = tokens['input_ids'].shape[1]
            return sentimento, float(score), num_tokens
        else:
            return ("Indefinido", 0.0, 0)
    except Exception as e:
        return ("Erro", 0.0, 0)

# App principal
def main():
    st.set_page_config(page_title="Ideias para Florianópolis", layout="wide")
    st.title("💬 Mapa de Ideias e Sentimentos sobre Florianópolis")

    # Inicializa session state de forma consistente
    if 'novas_ideias' not in st.session_state:
        st.session_state.novas_ideias = pd.DataFrame(columns=['IDEIA', 'sentimento', 'confianca', 'num_tokens', 'bairro', 'latitude', 'longitude'])
    if 'dados_processados' not in st.session_state:
        st.session_state.dados_processados = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None

    uploaded_file = st.file_uploader("📁 Faça upload de um CSV com a coluna 'IDEIA'", type=["csv"])

    # Botão de reprocessamento (aparece apenas quando há dados processados)
    if uploaded_file is not None and st.session_state.dados_processados is not None:
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("↻ Reprocessar Arquivo", 
                       help="Clique para processar novamente o arquivo CSV",
                       type="primary"):
                st.session_state.dados_processados = None
                st.session_state.novas_ideias = pd.DataFrame(columns=['IDEIA', 'sentimento', 'confianca', 'num_tokens', 'bairro', 'latitude', 'longitude'])
                st.rerun()

    # Processamento do CSV (apenas na primeira carga ou após reset)
    if uploaded_file is not None and st.session_state.dados_processados is None:
        with st.spinner('Processando arquivo CSV...'):
            df_temp = carregar_dados(uploaded_file)
            if 'IDEIA' in df_temp.columns:
                # Carrega modelo apenas se necessário
                if st.session_state.analyzer is None:
                    st.session_state.analyzer, st.session_state.tokenizer = carregar_modelo()
                
                # Processa os dados
                df_temp[['sentimento', 'confianca', 'num_tokens']] = df_temp['IDEIA'].astype(str).apply(
                    lambda x: pd.Series(analisar_sentimento_completo(x, st.session_state.analyzer, st.session_state.tokenizer)))
                
                df_temp['bairro'] = df_temp['IDEIA'].astype(str).apply(extrair_bairro)
                df_temp[['latitude', 'longitude']] = df_temp['bairro'].apply(
                    lambda x: pd.Series(geocodificar_bairro(x)) if pd.notna(x) else pd.Series([None, None]))
                
                st.session_state.dados_processados = df_temp
                st.success("Dados processados com sucesso!")
            else:
                st.error("O arquivo deve conter a coluna 'IDEIA'")

    # Seção para adicionar nova ideia
    with st.expander("➕ Adicionar Nova Ideia", expanded=True):
        with st.form("nova_ideia_form"):
            nova_ideia = st.text_area("Digite sua ideia, sugestão ou crítica sobre Florianópolis:", height=150)
            enviar = st.form_submit_button("Analisar Sentimento")
            
            if enviar and nova_ideia:
                # Garante que o modelo está carregado
                if st.session_state.analyzer is None:
                    st.session_state.analyzer, st.session_state.tokenizer = carregar_modelo()
                
                with st.spinner("Analisando nova ideia..."):
                    sentimento, confianca, num_tokens = analisar_sentimento_completo(
                        nova_ideia, st.session_state.analyzer, st.session_state.tokenizer)
                    bairro = extrair_bairro(nova_ideia)
                    latitude, longitude = geocodificar_bairro(bairro) if bairro else (None, None)
                    
                    nova_linha = pd.DataFrame([{
                        'IDEIA': nova_ideia,
                        'sentimento': sentimento,
                        'confianca': confianca,
                        'num_tokens': num_tokens,
                        'bairro': bairro,
                        'latitude': latitude,
                        'longitude': longitude
                    }])
                    
                    st.session_state.novas_ideias = pd.concat(
                        [st.session_state.novas_ideias, nova_linha], 
                        ignore_index=True
                    )
                    st.success("Ideia adicionada com sucesso!")
                    st.balloons()

    # Combina dados apenas para o mapa
    df_completo = pd.concat([
        st.session_state.dados_processados if st.session_state.dados_processados is not None else pd.DataFrame(),
        st.session_state.novas_ideias
    ], ignore_index=True)
    
    # Mostra mapa se houver dados
    if not df_completo.empty:
        mapa = folium.Map(location=[-27.5954, -48.5480], zoom_start=12)
        for _, row in df_completo.dropna(subset=['latitude', 'longitude']).iterrows():
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
        st_folium(mapa, width=1000, height=600, key="mapa")
    else:
        st.info("Nenhum dado disponível para exibir o mapa. Carregue um arquivo CSV ou adicione uma nova ideia.")

    # Tabelas separadas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Ideias do Arquivo CSV")
        if st.session_state.dados_processados is not None:
            st.dataframe(st.session_state.dados_processados[['IDEIA', 'sentimento', 'confianca', 'num_tokens', 'bairro']])
        else:
            st.info("Nenhum arquivo CSV carregado ainda.")

    with col2:
        st.subheader("✨ Novas Ideias Adicionadas")
        if not st.session_state.novas_ideias.empty:
            st.dataframe(st.session_state.novas_ideias[['IDEIA', 'sentimento', 'confianca', 'num_tokens', 'bairro']])
        else:
            st.info("Nenhuma nova ideia adicionada ainda.")

if __name__ == '__main__':
    main()

