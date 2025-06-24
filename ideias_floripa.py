
import pandas as pd
import folium
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

def main():
    # Caminho do CSV no GitHub
    caminho_arquivo = 'https://raw.githubusercontent.com/fabiospiazzi/ideias_floripa/main/dados_ideias_floripa.csv'

    # Carregar dados
    df = pd.read_csv(caminho_arquivo)

    # Modelo de sentimento
    model_name = "pierreguillou/bert-base-cased-sentiment-analysis-sst-2-portuguese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # Configuração da página
    PAGE_CONFIG = {"page_title":"Demandas de Ideias - Florianópolis", "page_icon":":smiley:", "layout":"centered"}
    st.set_page_config(**PAGE_CONFIG)

    # Análise de sentimento + contagem de tokens
    def analisar_sentimento_com_tokens(texto):
        try:
            tokens = tokenizer(texto, truncation=True, max_length=512, return_tensors=None)
            num_tokens = len(tokens['input_ids'][0])
            resultado = sentiment_analyzer(texto[:512])[0]
            return pd.Series([resultado['label'], float(resultado['score']), num_tokens])
        except Exception as e:
            return pd.Series(["Erro", 0.0, 0])

    df[['sentimento', 'confianca', 'num_tokens']] = df['IDEIA'].astype(str).apply(analisar_sentimento_com_tokens)

    # Lista de bairros de Florianópolis
    bairros_floripa = [
        "Centro", "Trindade", "Ingleses do Rio Vermelho", "Canasvieiras", "Rio Tavares",
        "Estreito", "Agronômica", "Capoeiras", "Itacorubi", "Campeche", "Córrego Grande",
        "Jurerê", "Costeira do Pirajubaé", "Lagoa da Conceição", "Saco dos Limões", "Pantanal",
        "Santa Mônica", "João Paulo", "Abraão", "Carianos", "Monte Verde", "Coqueiros",
        "Barra da Lagoa", "Tapera", "Ribeirão da Ilha", "Sambaqui", "Armação", "Ratones"
    ]

    # Função para extrair bairro mencionado
    def extrair_bairro(texto):
        texto_lower = texto.lower()
        for bairro in bairros_floripa:
            if re.search(rf"\b{re.escape(bairro.lower())}\b", texto_lower):
                return bairro
        return None

    df['bairro'] = df['IDEIA'].astype(str).apply(extrair_bairro)

    # Função para geocodificação
    geolocator = Nominatim(user_agent="floripa-sentimento-mapa")

    def geocodificar_bairro(bairro):
        try:
            local = geolocator.geocode(f"{bairro}, Florianópolis, Santa Catarina, Brasil")
            if local:
                return pd.Series([local.latitude, local.longitude])
        except GeocoderTimedOut:
            time.sleep(1)
            return geocodificar_bairro(bairro)
        return pd.Series([None, None])

    df[['latitude', 'longitude']] = df['bairro'].apply(lambda x: geocodificar_bairro(x) if pd.notna(x) else pd.Series([None, None]))

    # Criar mapa Folium
    mapa = folium.Map(location=[-27.5954, -48.5480], zoom_start=12)

    for _, row in df.dropna(subset=['latitude', 'longitude']).iterrows():
        cor = "green" if row['sentimento'] == "Positive" else "red"
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"<b>Bairro:</b> {row['bairro']}<br><b>Sentimento:</b> {row['sentimento']}<br><b>Confiança:</b> {row['confianca']:.2f}",
            icon=folium.Icon(color=cor)
        ).add_to(mapa)

    # Salvar mapa como HTML
    mapa.save("mapa_sentimentos_floripa.html")
    print("Mapa salvo como 'mapa_sentimentos_floripa.html'")

if __name__ == '__main__':
    main()
