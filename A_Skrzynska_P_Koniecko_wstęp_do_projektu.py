# WŁĄCZENIE KODU
# Aby kod zadziałał, należy wpisać w Terminalu komendę "pip install pandas numpy matplotlib seaborn scikit-learn openpyxl"
# Należy upewnić się, że plik "dane_wojewodztwa.xlsx" oraz "A_Skrzynska_P_Koniecko_wstęp_do_projektu.py" znajdują się w tym samym miejscu (na pulpicie)
# w Terminalu należy wpisać "cd Desktop"
# Następnie należy wpisać komendę "python "A_Skrzynska_P_Koniecko_wstęp_do_projektu.py" "
# Powinna pokazać się macierz korelacji w oddzielnych oknie i w celu zobaczenia następnych wykresów należy zamknąć to okno
# Pojawi się wtedy wykres osypiska i należy zamknąć to okno w celu pojawienia się rankingu końcowego


# Załadowanie wymaganych bibliotek
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# WCZYTYWANIE DANYCH
nazwa_pliku = 'dane_wojewodztwa.xlsx'

# Należy sprawdzić, czy pliki zapisane są w tym samym folderze
if not os.path.exists(nazwa_pliku):
    print(f"BŁĄD: Nie widzę pliku '{nazwa_pliku}'!")
    # Tworzymy przykładowe dane w celu sprawdzenia poprawności kodu
    print("Generuję dane zastępcze, żeby pokazać działanie programu")
    data = {'Wojewodztwo': [f'Woj_{i}' for i in range(1,17)]}
    for i in range(1, 17):
        data[f'X{i}'] = np.random.rand(16) * 100
    df = pd.DataFrame(data)
else:
    df_temp = pd.read_excel(nazwa_pliku, header=None)
    
    try:
        # Szukamy wiersza, w którym zaczyna się tabela (nagłówek X1)
        header_row = df_temp[df_temp.eq("X1").any(axis=1)].index[0]
        df = pd.read_excel(nazwa_pliku, skiprows=header_row)
    except IndexError:
        print("Nie znalazłem nagłówka 'X1', wczytuję standardowo od pierwszego wiersza.")
        df = pd.read_excel(nazwa_pliku)

# Bierzemy nazwy województw z pierwszej kolumny
wojewodztwa = df.iloc[:, 0]

# ANALIZA KORELACJI
zmienne_wszystkie = [col for col in df.columns if col.startswith('X')]
data_corr = df[zmienne_wszystkie].copy()

plt.figure(figsize=(12, 10))
# Heatmapa to mapa ciepła - czerwone pola to silne powiązania
sns.heatmap(data_corr.corr(), annot=True, cmap='RdYlGn', fmt=".2f", linewidths=0.5)
plt.title("Macierz korelacji wszystkich zmiennych (X1 - X16)")
plt.show()

# WYBÓR ZMIENNYCH DO PCA
# Zgodnie z założeniami projektu, do redukcji wymiarów wybieramy 9 zmiennych:
wybrane_pca = ["X1", "X2", "X4", "X6", "X7", "X10", "X11", "X12", "X15"]

# Sprawdzenie czy wybrane kolumny istnieją w pliku
dostepne_kolumny = [c for c in wybrane_pca if c in df.columns]
data_pca = df[dostepne_kolumny].copy()

# Destymulanta (X15 - Przestępczość)
# Zmieniamy znak, aby wysoka wartość PC1 oznaczała "lepiej"
if "X15" in data_pca.columns:
    print("Obracam zmienną X15 (przestępczość), bo to destymulanta.")
    data_pca["X15"] = -data_pca["X15"]

# STANDARYZACJA
# Standaryzacja sprowadza wartości własne do skali porównywalnej z Kryterium Kaisera
# StandardScaler zamienia wszystko na "odchylenia od średniej"

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_pca)

# OBLICZANIE PCA
pca = PCA()
pca_results = pca.fit_transform(data_scaled)
eigenvalues = pca.explained_variance_ # Wartości własne (siła składowej)

# Obliczanie procentowej wariancji
prop_var = pca.explained_variance_ratio_ * 100
cum_var = np.cumsum(prop_var)

print("\nANALIZA WYJAŚNIONEJ ZMIENNOŚCI:")
print(f"PC1 wyjaśnia: {prop_var[0]:.2f}% zmienności")
print(f"PC2 wyjaśnia: {prop_var[1]:.2f}% zmienności")
print(f"PC3 wyjaśnia: {prop_var[2]:.2f}% zmienności")
print(f"Łącznie te trzy składowe wyjaśniają: {cum_var[2]:.2f}% zmienności")

# WYKRES OSYPISKA (Scree Plot)
# Wykres osypiska pomaga zdecydować, ile składowych (PC) brać pod uwagę
# Na podstawie kryterium Kaisera wybieramy zmienne, które są powyżej czerwonej linii (Lambda > 1)
# Oznacza to, że wyjaśniają one największy procent wariancji

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(eigenvalues)+1), eigenvalues, color='skyblue', alpha=0.7, label='Wartość własna')
plt.plot(range(1, len(eigenvalues)+1), eigenvalues, marker='o', color='darkblue', linewidth=2)

# Kryterium Kaisera (Lambda = 1)
plt.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Kryterium Kaisera (λ=1)')

plt.title("Wykres osypiska (Scree Plot) dla wybranych zmiennych")
plt.xlabel("Numer Głównej Składowej (PC)")
plt.ylabel("Wartość własna (Eigenvalue)")
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()

# RANKING KOŃCOWY
# Tworzymy tabelę wynikową na podstawie pierwszej składowej (PC1)
ranking = pd.DataFrame({
    'Województwo': wojewodztwa,
    'Wynik_PCA': pca_results[:, 0]
})

# Jeżeli większość wyników jest ujemna, odwracamy je
if ranking['Wynik_PCA'].mean() < 0:
    ranking['Wynik_PCA'] = -ranking['Wynik_PCA']

ranking = ranking.sort_values(by='Wynik_PCA', ascending=False)
ranking = ranking.reset_index(drop=True)
ranking.index += 1 # Aby uzyskany ranking rozpoczynał się od 1 a nie od 0

print("\nOstateczny ranking atrakcyjności na podstawie PC1:")
print(ranking.to_string())