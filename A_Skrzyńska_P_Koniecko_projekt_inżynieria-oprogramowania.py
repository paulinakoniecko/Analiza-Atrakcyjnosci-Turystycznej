# WŁĄCZENIE KODU
# Aby kod zadziałał, należy wpisać w Terminalu komendę "pip install gradio plotly pandas numpy scikit-learn requests"
# Po zainstalowaniu wymaganych bibliotek należy uruchomić program w Terminalu wpisując "python "A_Skrzyńska_P_Koniecko_projekt_inżynieria oprogramowania.py" "
# Aby komenda uruchamiająca program zadziałała należy upewnić się, gdzie został zapisany plik (najprostszym sposobem będzie wpisanie "python (ze spacją) i przeciągnąć plik z pulpitu/folderu, w którym się znajduje (w taki sposób Terminal sam wpisze ścieżkę)" i nacisnąć enter)
# Terminal powinien pokazać link "Running on local URL:..." i należy ten link otworzyć w przeglądarce i nacisnąć przycisk "Analizuj"
# Po wykonaniu wszystkich działań w oknie przeglądarki (tej z linku) powinna pokazać się mapa, na którą można najechać, aby zobaczyć dokładne wyniki dla danego województwa


# Załadowanie wymaganych bibliotek
import requests
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import gradio as gr

# KONFIGURACJA
BASE = "https://bdl.stat.gov.pl/api/v1"
# W tym miejscu należy zarejestrować się w Banku Danych Lokalnych GUSu, aby uzyskać klucz w celu pobrania wszystkich zmiennych
MY_API_KEY = "d4a9b4b9-20f6-46cd-ae1e-08de56c2a7a2" 

# Nagłówki wysyłane z każdym zapytaniem. 'User-Agent' udaje przeglądarkę, 'X-ClientId' to autoryzacja w GUS
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'X-ClientId': MY_API_KEY
}

INDICATORS = {
    "Powierzchnia_terenow_chronionych": ["1540"],
    "Zielen_parki": ["73847", "73849", "73848"],
    "Miejsca_noclegowe": ["40936"],
    "Gastronomia_ogolem": ["64542", "64543", "64544", "64545"],
    "Liczba_imprez": ["377328"],
    "Dlugosc_komunikacji_miejskiej": ["400294"],
    "Dlugosc_drog_twardych": ["54574"],
    "Dlugosc_linii_kolejowych": ["4737"],
    "Liczba_przestepstw": ["58559"],
    "Populacja": ["72305"],  # Potrzebne do normalizacji (per capita)
    "Powierzchnia": ["1"]    # Potrzebne do normalizacji (na km2)
}

# GeoJSON
# GeoJSON to format pliku opisujący kształty geograficzne (granice województw)
# Jest niezbędny, aby Plotly wiedziało, jak narysować kontury na mapie
geojson_url = "https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/wojewodztwa/wojewodztwa-max.geojson"
try:
    geojson_poland = requests.get(geojson_url).json()
except:
    geojson_poland = {}
    print("Błąd pobierania GeoJSON")

def fetch_bulk_data(year):
    data_store = {}
    print(f"Rozpoczynam pobieranie danych za rok {year}...")

    for indicator_name, var_ids in INDICATORS.items():
        print(f"Pobieranie wskaźnika: {indicator_name}")
        for vid in var_ids:
            # unit-level=2 oznacza poziom województw, page-size=100 pobiera wszystkie województwa w jednym zapytaniu
            url = f"{BASE}/data/by-variable/{vid}?unit-level=2&year={year}&format=json&page-size=100"
            success = False

            # Mechanizm RETRY: Próbujemy 5 razy w razie błędu sieci lub limitu zapytań
            for _ in range(5):
                try:
                    r = requests.get(url, headers=HEADERS, timeout=30)
                    if r.status_code == 200:
                        data = r.json()
                        for item in data.get("results", []):
                            unit_id = item["id"]
                            unit_name = item["name"]
                            # Pobieramy wartość, a jeśli jej brak (None), wstawiamy 0.0
                            val = float(item["values"][0]["val"]) if item.get("values") else 0.0

                            if unit_id not in data_store:
                                data_store[unit_id] = {"Województwo": unit_name}
                                for k in INDICATORS.keys(): data_store[unit_id][k] = 0.0

                            # Sumowanie ze względu na występowanie np. zmiennej "Zielen_parki" (suma 3 różnych zmiennych z API)
                            data_store[unit_id][indicator_name] += val
                        success = True
                        break
                    elif r.status_code == 429:
                        # Kod 429 oznacza "Too Many Requests" - dane nie zostaną pobrane ze względu na zbyt szybki czas poboru
                        time.sleep(5)
                    else:
                        time.sleep(1)
                except:
                    time.sleep(2)

            # Krótka pauza między zapytaniami, aby serwer API pozwolił na zebranie danych
            time.sleep(0.2)

    return pd.DataFrame(data_store.values())

def analyze_tourism(year: str):
    # Zabezpieczenie w sytuacji, gdy użytkownik wpisze spacje lub nic, domyślnie 2022
    year = str(year).strip() or "2022"
    df = fetch_bulk_data(year)
    if df.empty: return pd.DataFrame(), None #  Zwracamy puste wyniki w razie błędu pobierania
    
    df_display = df.copy()
    
    # PRZETWARZANIE
    # Zabezpieczenie przed dzieleniem przez zero
    df["Powierzchnia"] = df["Powierzchnia"].replace(0, 1)
    df["Populacja"] = df["Populacja"].replace(0, 1)
    
    # Tworzenie wskaźników względnych (X1, X2...):
    df["X1"] = df["Powierzchnia_terenow_chronionych"] / df["Powierzchnia"]
    df["X2"] = df["Zielen_parki"] / df["Powierzchnia"]
    df["X4"] = df["Miejsca_noclegowe"] / df["Populacja"] * 1000
    df["X6"] = df["Gastronomia_ogolem"] / df["Populacja"] * 10000
    df["X7"] = df["Liczba_imprez"] / 365
    df["X10"] = df["Dlugosc_komunikacji_miejskiej"] / df["Populacja"] * 10000
    df["X11"] = df["Dlugosc_drog_twardych"] / df["Populacja"] * 1000
    df["X12"] = df["Dlugosc_linii_kolejowych"] / df["Populacja"] * 10000
    
    # Destymulanta
    # Przestępstwo jest zjawiskiem negatywny, dlatego aby użyć go w rankingu atrakcyjność należy odwrócić działanie tej zmiennej
    sum_przest = df["Liczba_przestepstw"].sum() or 1
    df["X15"] = - (df["Liczba_przestepstw"] / sum_przest * 100)

    # Lista kolumn, które biorą udział w finalnym rankingu
    cols = ["X1","X2","X4","X6","X7","X10","X11","X12","X15"]

    # Usuwamy nieskończoności (np. dzielenie przez 0) i braki danych
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # STANDARYZACJA
    X = StandardScaler().fit_transform(df[cols])

    # OBLICZENIE PCA
    pca = PCA().fit(X)
    scores = pca.transform(X)
    if sum(pca.components_[0]) < 0: scores[:,0] = -scores[:,0]

    # Przygotowanie ramki danych do mapy
    df_map = pd.DataFrame({"Województwo": df["Województwo"], "PC1": scores[:,0]})
    # Normalizacja nazwy do małych liter, aby pasowała do klucza w GeoJSON
    df_map["geo"] = df_map["Województwo"].str.lower()

    # RYSOWANIE MAPY
    fig = px.choropleth(df_map, geojson=geojson_poland, locations="geo", # Locations - kolumna z nazwą województw
                        featureidkey="properties.nazwa", color="PC1",    # Featureidkey - ścieżka do nazwy woj. w pliku GeoJSON
                        color_continuous_scale="RdYlGn",                 # Skala: Czerwony (nisko) -> Żółty -> Zielony (wysoko)
                        hover_name="Województwo")                        # Co wyświetlić po najechaniu myszką
    # Dopasowanie widoku mapy do danych
    fig.update_geos(fitbounds="locations", visible=False)
    
    return df_display, fig

# INTERFJS UŻYTKOWNIKA (gradio)
with gr.Blocks() as demo:
    gr.Markdown("## Analiza Turystyczna")
    with gr.Row():
        year_input = gr.Textbox(label="Rok", value="2022")
        btn = gr.Button("Analizuj")
    data_out = gr.Dataframe()
    plot_out = gr.Plot()
    # Podpięcie funkcji pod przycisk
    btn.click(fn=analyze_tourism, inputs=year_input, outputs=[data_out, plot_out])

# Uruchomienie serwera lokalnego
if __name__ == "__main__":
    demo.launch()