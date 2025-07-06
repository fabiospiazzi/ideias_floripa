# app.py

import streamlit as st
import pandas as pd
import re
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

#############################################################CONFIGURAÇÃO INICIAL##############################################################################
# Serve para manter o estado do código e não ficar atualizando toda hora que há uma interação com o resto do app.
if 'dados_processados' not in st.session_state:
    st.session_state.dados_processados = None
#############################################################CARREGA O ARQUIVO CSV#####################################################################################
# Função para carregar CSV por upload
@st.cache_data #armazena dados no cache pra não ficar carregando toda hora, a não ser se um novo arquivo for carregado
def carregar_dados(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file) #A primeira vez, verifica se o upload foi feito, se sim, joga pro pandas (pd)
    else:
        st.info("Nenhum arquivo CSV carregado ainda.")
        #Aqui é um teste - carrega o banco de dados direto to github para testes - NÃO USADO
        #url = "https://raw.githubusercontent.com/fabiospiazzi/ideias_floripa/main/dados_ideias_floripa.csv"
        #return pd.read_csv(url)
#############################################################LISTA DE COMPARAÇÃO DE BAIRROS##############################################################################
# Lista fixa de bairros
bairros_floripa = [
    "Vargem Pequena", "Vargem Grande", "Vargem do Bom Jesus", "Vargem de Fora", "Trindade",
    "Tapera da Base", "Tapera", "Santo Antônio", "Santinho", "Santa Mônica", "Sambaqui",
    "Saco Grande", "Saco dos Limões", "Rio Vermelho", "Rio Tavares do Norte",
    "Rio Tavares Central", "Rio Tavares", "Rio das Pacas", "Ribeirão da Ilha", "Retiro", "Ressacada",
    "Recanto dos Açores", "Ratones", "Praia Mole", "Praia Brava", "Porto da Lagoa",
    "Ponta das Canas", "Pedrita", "Pântano do Sul", "Pantanal", "Morro do Peralta",
    "Morro das Pedras", "Monte Verde", "Monte Cristo", "Moenda", "Lagoinha do Norte",
    "Lagoa Pequena", "Lagoa", "Jurere Oeste", "Jurere Leste", "Jurerê", "José Mendes",
    "João Paulo", "Jardim Atlântico", "Itaguaçu", "Itacorubi", "Ingleses Sul",
    "Ingleses Norte", "Ingleses Centro", "Ingleses", "Forte", "Estreito", "Dunas da Lagoa",
    "Daniela", "Costeiro do Ribeirão", "Costeira do Pirajubaé", "Córrego Grande",
    "Coqueiros", "Coloninha", "Centro", "Carianos", "Capoeiras", "Capivari",
    "Canto dos Araçás", "Canto do Lamim", "Canto da Lagoa", "Canto", "Canasvieiras", "Campeche",
    "Campeche Sul", "Campeche Norte", "Campeche Leste", "Campeche Central",
    "Caieira", "Caiacanga", "Cacupé", "Cachoeira do Bom Jesus Leste",
    "Cachoeira do Bom Jesus", "Bom Abrigo", "Base Aérea", "Barra do Sambaqui",
    "Barra da Lagoa", "Balneário", "Autódromo", "Armação", "Alto Ribeirão Leste",
    "Alto Ribeirão", "Agronômica", "Açores", "Abraão"
]

#############################################################EXTRAIR OA BAIRROS DO TEXTO##############################################################################
def extrair_bairro(texto):
    texto_lower = texto.lower() #passa o texto para minusculo
    for bairro in bairros_floripa:
        if re.search(rf"\b{re.escape(bairro.lower())}\b", texto_lower): #lopping para buscar os bairros, usa \b para buscar palasvras inteiras e re.escape trata os caracteres especiais
            return bairro
    return None
#############################################################GEOLOCALIZA OS BAIRROS###################################################################################
# Geolocalizador
geolocator = Nominatim(user_agent="floripa-sentimento-streamlit") #nome de usuário para usar a API, pode ser qualquer um que identifique o projeto

@st.cache_data # só geocodifica para aquele bairro uma vez
def geocodificar_bairro(bairro):
    try:
        local = geolocator.geocode(f"{bairro}, Florianópolis, SC, Brasil", timeout=10) #procura no openstreet map o nome do bairro mais Florianópois mais Brasil a cada 10 s
        if local:
            return (local.latitude, local.longitude) #Se acha, retorna latitude e longitude
    except GeocoderTimedOut: #tenta conectar a cada 2 segundos, geralmente conecta.
        time.sleep(2)
        return geocodificar_bairro(bairro)
    return (None, None) #Se não acha o bairro, retorna None, None nas coordenadas.
#############################################################CARREGA O MODELO BERT####################################################################################
# Carrega modelo e tokenizer
@st.cache_resource #mantém o modelo carregado na memória
def carregar_modelo():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment" #Modelo usado
    tokenizer = AutoTokenizer.from_pretrained(model_name) #pré-processamento do modelo
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer), tokenizer

#############################################################FAZ A ANÁLISE DE SENTIMENTO################################################################################
# Função para analisar sentimento
def analisar_sentimento_completo(texto, sentiment_analyzer, tokenizer):
    try:
        resultado = sentiment_analyzer(texto[:512])[0] #Analisa somente 512 token. É uma limitação dos modelos BERT
        label = resultado.get('label', '')
        score = resultado.get('score', 0.0)
        
        match = re.match(r"(\d)\s*stars?", label, re.IGNORECASE) #retorna o número de estrelas atribuito na análise de sentimento
        if match:
            estrelas = int(match.group(1)) # joga pra dentro de estrela o valor da qtd de estrelas
            sentimento = (
                "Negativo" if estrelas <= 2 else
                "Neutro" if estrelas == 3 else
                "Positivo"
            )
            tokens = tokenizer(texto, truncation=True, max_length=512, return_tensors="pt")
            num_tokens = tokens['input_ids'].shape[1] #quantidade de tokens utlizado na frase
            return sentimento, float(score), num_tokens
        else:
            return ("Indefinido", 0.0, 0)
    except Exception as e:
        return ("Erro", 0.0, 0)
#############################################################APP PRINCIPAL################################################################################
# App principal
def main():
    st.set_page_config(page_title="Ideias para Florianópolis", layout="wide") #Texto que vai na haba do navegador
    st.title("💬 Mapa de Ideias e Sentimentos sobre Florianópolis") #Título da página

    # Inicialização do session state pra não ficar autualizando automático toda hora que mexe no mapa
    if 'dados_processados' not in st.session_state:
        st.session_state.dados_processados = None
    if 'novas_ideias' not in st.session_state:
        st.session_state.novas_ideias = pd.DataFrame(columns=['IDEIA', 'sentimento', 'confianca', 'num_tokens', 'bairro', 'latitude', 'longitude'])
    if 'arquivo_carregado' not in st.session_state:
        st.session_state.arquivo_carregado = False
    if 'texto_ideia' not in st.session_state:
        st.session_state.texto_ideia = ""

    uploaded_file = st.file_uploader("📁 Faça upload de um CSV com a coluna 'IDEIA'", type=["csv"]) #Widget pra fazer o carregamento de um aquivo

    # Botão de processamento (só aparece se fizer o upload de um aquivo e arquivo ainda não foi processado)##################################################
    if uploaded_file is not None: #and not st.session_state.arquivo_carregado:
        if st.button("⚙️Processar Arquivo", type="primary"):
            # Inicia o container de progresso
            progress_bar = st.progress(0)
            status_text = st.empty()  # Para mensagens dinâmicas
    
            with st.spinner('Processando arquivo CSV...'):
                # Reseta o estado se já existirem dados processados
                st.session_state.dados_processados = None
                st.session_state.arquivo_carregado = False
                
                df_temp = carregar_dados(uploaded_file)
                
                if 'IDEIA' in df_temp.columns:
                    # Passo 1/4: Carregar modelo (10%)
                    status_text.text("Carregando modelo...")
                    progress_bar.progress(10)
                    sentiment_analyzer, tokenizer = carregar_modelo()
                    
                    # Passo 2/4: Análise de sentimentos (40%)
                    status_text.text("Analisando sentimentos...")
                    resultados = []
                    for i, texto in enumerate(df_temp['IDEIA'].astype(str)):
                        resultados.append(analisar_sentimento_completo(texto, sentiment_analyzer, tokenizer))
                        progress_bar.progress(10 + int(30 * (i + 1) / len(df_temp)))
                    
                    df_temp[['sentimento', 'confianca', 'num_tokens']] = resultados
                    progress_bar.progress(40)
                    
                    # Passo 3/4: Extrair bairros (70%)
                    status_text.text("Identificando bairros...")
                    df_temp['bairro'] = df_temp['IDEIA'].astype(str).apply(extrair_bairro)
                    progress_bar.progress(70)
                    
                    # Passo 4/4: Geocodificação (100%)
                    status_text.text("Geolocalizando...")
                    coordenadas = []
                    for i, bairro in enumerate(df_temp['bairro']):
                        coordenadas.append(geocodificar_bairro(bairro) if pd.notna(bairro) else (None, None))
                        progress_bar.progress(70 + int(30 * (i + 1) / len(df_temp)))
                    
                    df_temp[['latitude', 'longitude']] = coordenadas
                    
                    # Finalização
                    st.session_state.dados_processados = df_temp
                    st.session_state.analyzer = sentiment_analyzer
                    st.session_state.tokenizer = tokenizer
                    st.session_state.arquivo_carregado = True
                    
                    progress_bar.progress(100)
                    status_text.text("Processamento completo!")
                    st.success("Dados processados com sucesso!")
                    
                else:
                    progress_bar.empty()
                    st.error("O arquivo deve conter a coluna 'IDEIA'")

    # Seção para adicionar nova ideia (sempre visível)
    with st.expander("➕ Adicionar Nova Ideia", expanded=True):
        with st.form("nova_ideia_form"):
            nova_ideia = st.text_area(
                "Digite sua ideia, sugestão ou crítica sobre Florianópolis:",
                height=150,
                value=st.session_state.texto_ideia,
                key="input_ideia"
            )
            col1, col2 = st.columns([4, 1])
            with col2:
                enviar = st.form_submit_button("📨Enviar para Análise")
            with col1:
                if st.form_submit_button("Limpar Texto"):
                    st.session_state.texto_ideia = " "
                    st.rerun()
            
            if enviar and nova_ideia:
                if 'analyzer' not in st.session_state or st.session_state.analyzer is None:
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
                    if bairro is None:
                        st.warning("⚠️ Não foi possível identificar um bairro válido na demanda. O mapa/marcador não será exibido para este registro.")
                    else:
                        st.success("Ideia/Sugestão adicionada com sucesso! - Bairro Localizado!")
                        #st.balloons()

    # Botão para limpar todos os dados
    if st.session_state.dados_processados is not None or not st.session_state.novas_ideias.empty:
        if st.button("🧹 Limpar Todos os Dados", type="secondary"):
            st.session_state.dados_processados = None
            st.session_state.novas_ideias = pd.DataFrame(columns=['IDEIA', 'sentimento', 'confianca', 'num_tokens', 'bairro', 'latitude', 'longitude'])
            st.session_state.arquivo_carregado = False
            st.session_state.analyzer = None
            st.session_state.tokenizer = None
            st.session_state.texto_ideia = ""
            st.rerun()
            st.success("Todos os dados foram removidos!")

    # Combina dados apenas para o mapa
    df_completo = pd.concat([
        st.session_state.dados_processados if st.session_state.dados_processados is not None else pd.DataFrame(),
        st.session_state.novas_ideias
    ], ignore_index=True)
    
    # Mostra mapa se houver dados
    if not df_completo.empty:
        # Filtra apenas registros com geolocalização válida
        df_com_mapa = df_completo.dropna(subset=['latitude', 'longitude'])
        if not df_com_mapa.empty:
            st.subheader("🗺️ Mapa com Ideias Geolocalizadas")
            mapa = folium.Map(location=[-27.5954, -48.5480], zoom_start=12)
            marker_cluster = MarkerCluster().add_to(mapa)
            for _, row in df_com_mapa.iterrows():
                cor = (
                    "green" if row['sentimento'] == "Positivo" else
                    "blue" if row['sentimento'] == "Neutro" else
                    "red"
                )
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=f"<b>Ideia:</b> {row['IDEIA'][:100]}...<b>Bairro:</b> {row['bairro']}<br><b>Sentimento:</b> {row['sentimento']}<br><b>Confiança:</b> {row['confianca']:.2f}",
                    icon=folium.Icon(color=cor)
                ).add_to(marker_cluster)
            st_folium(mapa, width=1000, height=600, key="mapa")
        else:
            st.warning("Nenhuma demanda com geolocalização válida para exibir no mapa.")
    else:
        st.info("Nenhum dado disponível para exibir o mapa. Carregue um arquivo CSV ou adicione uma nova ideia.")

    # Tabelas separadas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Ideias do Arquivo")
        if st.session_state.dados_processados is not None:
            st.dataframe(st.session_state.dados_processados[['IDEIA', 'sentimento', 'confianca', 'num_tokens', 'bairro']])
        else:
            st.info("Nenhum arquivo processado ainda.")

    with col2:
        st.subheader("✨ Novas Ideias Adicionadas")
        if not st.session_state.novas_ideias.empty:
            st.dataframe(st.session_state.novas_ideias[['IDEIA', 'sentimento', 'confianca', 'num_tokens', 'bairro']])
        else:
            st.info("Nenhuma nova ideia adicionada ainda.")

if __name__ == '__main__':
    main()

