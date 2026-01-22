# Dokumentacja projektu - Inżynieria Oprogramowania 2025/2026

## Spis treści
1. [Charakterystyka oprogramowania](#1-charakterystyka-oprogramowania)
2. [Prawa autorskie](#2-prawa-autorskie)
3. [Specyfikacja wymagań](#3-specyfikacja-wymagań)
4. [Architektura systemu/oprogramowania](#4-architektura-systemuoprogramowania)
5. [Struktura plików projektu](#5-struktura-plików-projektu)
6. [Metodologia: Analiza wstępna i dobór zmiennych](#6-metodologia-analiza-wstępna-i-dobór-zmiennych)
7. [Instrukcja uruchomienia](#7-instrukcja-uruchomienia)
8. [Testy i sprawozdanie](#8-testy-i-sprawozdanie)

---

## 1. Charakterystyka oprogramowania

**a. Nazwa skrócona:** Turystyka-PCA

**b. Nazwa pełna:** System analityczny do oceny atrakcyjności turystycznej województw Polski z wykorzystaniem wskaźników kompozytowych oraz metod statystycznej analizy wielowymiarowej.

**c. Sumaryczny opis ze wskazaniem celów:** Oprogramowanie służy do automatycznego pobierania, przetwarzania i wizualizacji danych statystycznych z API Banku Danych Lokalnych GUS. Głównym celem systemu jest konstrukcja obiektywnego rankingu atrakcyjności regionów przy użyciu Analizy Głównych Składowych (PCA). Narzędzie przekształca wielowymiarowy zbiór 11 wskaźników w ujednolicony **Regionalny Indeks Atrakcyjności Turystycznej (RIAT)**, prezentowany w formie interaktywnej mapy.

---

## 2. Prawa autorskie

**a. Autorzy:** Paulina Koniecko, Alicja Skrzyńska  
**b. Warunki licencyjne:** Oprogramowanie udostępniane jest na licencji **MIT**. Pozwala ona na dowolne wykorzystanie kodu pod warunkiem zachowania informacji o autorach.

---

## 3. Specyfikacja wymagań

### Legenda kategorii i priorytetów
* **Priorytet 1:** Wymagane (krytyczne dla działania systemu).
* **Priorytet 2:** Przydatne (rozszerzające funkcjonalność).
* **Kategoria Funkcjonalna (F):** Opisuje funkcje realizowane przez kod.
* **Kategoria Pozafunkcjonalna (NF):** Opisuje atrybuty techniczne i jakościowe.
* **Status Met:** Wymaganie zrealizowane, zweryfikowane i obecne w kodzie.

### Moduł 1: Pobieranie i Przetwarzanie Danych
| Identyfikator | Nazwa | Opis | Priorytet | Kategoria | Status |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **F1.1** | Pobieranie API | Pobieranie danych dla 11 wskaźników metodą `by-variable`. | 1 | F | **Met** |
| **F1.2** | Normalizacja | Przeliczanie danych per capita oraz na jednostki powierzchni (km²). | 1 | F | **Met** |
| **NF1.1** | Obsługa błędów | Mechanizm Retry Logic dla błędów sieciowych i limitów API (429). | 1 | NF | **Met** |

### Moduł 2: Analiza Statystyczna i UI
| Identyfikator | Nazwa | Opis | Priorytet | Kategoria | Status |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **F2.1** | Analiza PCA | Wyliczanie składowej PC1 (wskaźnik RIAT) na bazie danych. | 1 | F | **Met** |
| **F2.2** | Mapa | Generowanie interaktywnego kartogramu Polski (Plotly). | 1 | F | **Met** |
| **F2.3** | Interfejs | Udostępnienie pola wyboru roku danych w GUI Gradio. | 2 | F | **Met** |

---

## 4. Architektura systemu/oprogramowania

**a. Architektura rozwoju:** Python 3.12, Visual Studio Code, GitHub, Anaconda/Pip.

**b. Architektura uruchomieniowa:**
| Technologia | Przeznaczenie | Wersja |
| :--- | :--- | :--- |
| **Requests** | Obsługa zapytań HTTP do API GUS | 2.32.3 |
| **Pandas / Numpy** | Manipulacja ramkami danych i macierzami | 2.2.2 / 1.26.4 |
| **Scikit-learn** | Obliczenia statystyczne (PCA) i standaryzacja | 1.5.1 |
| **Plotly** | Generowanie interaktywnych map i wykresów | 5.24.1 |
| **Gradio** | Serwer interfejsu webowego (User Interface) | 6.3.0 |

**c. Prezentacja omawiająca wykorzystywane technologie:** Pełna prezentacja merytoryczna omawiająca wskaźniki kompozytowe oraz technologie znajduje się w pliku: [Wskaźniki-kompozytowe_PK_AS.pdf](./Wskaźniki-kompozytowe_PK_%20AS.pdf)

---

## 5. Struktura plików projektu
* `Alicja_Skrzynska_Paulina_Koniecko_projekt_inzynieria_oprogramowania.ipynb` – Główna aplikacja analityczna z interfejsem.
* `Analiza_Wstepna_Wybor_Zmiennych.ipynb` – Kod dokumentujący proces selekcji zmiennych (korelacje, osypisko).
* `dane_wojewodztwa.xlsx` – Dane historyczne użyte do kalibracji modelu PCA.
* `requirements.txt` – Lista bibliotek niezbędnych do uruchomienia.

---

## 6. Metodologia: Analiza wstępna i dobór zmiennych

W procesie budowy wskaźnika kompozytowego kluczowe było zachowanie obiektywizmu statystycznego. Proces ten udokumentowano w analizie wstępnej:
* **Selekcja zmiennych:** Wybrano 11 wskaźników (stymulanty i destymulanty) opisujących potencjał turystyczny regionów.
* **Analiza korelacji:** Za pomocą macierzy korelacji wyeliminowano zmienne nadmiarowe, pozostawiając 9 kluczowych cech do analizy PCA.
* **Wykres osypiska (Scree Plot):** Potwierdził on zasadność redukcji wymiarów. Analiza wykazała, że:
    * **PC1** wyjaśnia **48,86%** zmienności (główny wymiar atrakcyjności).
    * **PC2** wyjaśnia **18,00%** zmienności.
    * **PC3** wyjaśnia **13,69%** zmienności.
    * Łącznie pierwsze trzy składowe tłumaczą aż **80,54%** całkowitej zmienności zbioru danych.
* **Destymulanty:** Zmienna "Liczba przestępstw" (X15) została wprowadzona z odwróconym znakiem, dzięki czemu wysoka wartość wskaźnika syntetycznego zawsze oznacza pozytywny aspekt (wyższe bezpieczeństwo).

---

## 7. Instrukcja uruchomienia

### Krok 1: Uzyskanie klucza API
1.  Wejdź na stronę [BDL API GUS](https://api.stat.gov.pl/Home/BdlApi).
2.  Zarejestruj się i wygeneruj własny klucz API (identyfikator klienta).
3.  Otwórz kod aplikacji i wklej klucz w miejscu: `MY_API_KEY = "TWÓJ_KLUCZ"`.

### Krok 2: Instalacja środowiska
Uruchom w terminalu systemowym lub wewnątrz środowiska wirtualnego:
```bash
pip install requests pandas numpy scikit-learn plotly gradio
```

### Krok 3: Uruchomienie aplikacji
1. Otwórz główny plik projektu `Alicja_Skrzynska_Paulina_Koniecko_projekt_inzynieria_oprogramowania.ipynb` w środowisku Jupyter Notebook lub VS Code.
2. Uruchom wszystkie komórki za pomocą opcji **Run All**.
3. Po zainicjowaniu interfejsu, wybierz interesujący Cię rok w panelu Gradio i kliknij przycisk **"Pobierz dane i generuj mapę"**.

---

## 8. Testy i sprawozdanie

### Krok 1: Definicja scenariuszy testowych
Poniższa tabela przedstawia kluczowe przypadki testowe mające na celu weryfikację stabilności i poprawności algorytmu:

| ID Testu | Nazwa | Opis scenariusza | Oczekiwany rezultat |
| :--- | :--- | :--- | :--- |
| **ST_01** | **Test kompletności** | Pobranie danych dla 16 województw za pełny rok kalendarzowy. | Ramka danych (DataFrame) zawiera dokładnie 16 rekordów bez brakujących wartości (NaN). |
| **ST_02** | **Test odporności sieciowej** | Symulacja błędu 429 (Too Many Requests) podczas odpytywania API. | System przechwytuje błąd, wyświetla komunikat o oczekiwaniu (Retry Logic) i kontynuuje pracę po przerwie. |
| **ST_03** | **Test wizualizacji (UI)** | Najechane kursorem na obszar mapy po wygenerowaniu wyników. | Wyświetlenie interaktywnego okna (tooltip) z nazwą województwa i wartością RIAT. |


### Krok 2: Wykonanie testów i sprawozdanie
Wszystkie testy zostały przeprowadzone w środowisku produkcyjnym aplikacji. Poniżej przedstawiono wyniki:

* **ST_01 (Wynik pozytywny):** Dane pobrane pomyślnie. Potwierdzono spójność danych dla każdego z 16 regionów Polski.
* **ST_02 (Wynik pozytywny):** Mechanizm `Retry Logic` zadziałał poprawnie. Skrypt nie przerwał pracy po wystąpieniu limitu zapytań i dokończył analizę.
* **ST_03 (Wynik pozytywny):** Interfejs Gradio poprawnie wyrenderował mapę Plotly. Dane są czytelne, a interakcja płynna.
