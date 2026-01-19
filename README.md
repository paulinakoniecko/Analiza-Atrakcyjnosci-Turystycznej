# Dokumentacja projektu - Inżynieria Oprogramowania 2025/2026

## Spis treści
1. [Charakterystyka oprogramowania](#1-charakterystyka-oprogramowania)
2. [Prawa autorskie](#2-prawa-autorskie)
3. [Specyfikacja wymagań](#3-specyfikacja-wymagań).
4. [Architektura systemu/oprogramowania](#4-architektura-systemuoprogramowania)
5. [Testy](#5-testy)

---
## 1. Charakterystyka oprogramowania
**a. Nazwa skrócona:** Turystyka-PCA

**b. Nazwa pełna:** System analityczny do oceny atrakcyjności turystycznej województw Polski z wykorzystaniem metod statystycznych i wizualizacji kartograficznej.

**c. Sumaryczny opis ze wskazaniem celów:** Oprogramowanie służy do automatycznego pobierania, przetwarzania i wizualizacji danych statystycznych z API Banku Danych Lokalnych GUS. Głównym celem systemu jest obiektywna ocena atrakcyjności turystycznej regionów przy użyciu Analizy Głównych Składowych (PCA). Narzędzie przekształca wielowymiarowy zbiór wskaźników (np. baza noclegowa, gastronomia, bezpieczeństwo) w ujednolicony ranking RIAT (Regionalny Indeks Atrakcyjności Turystycznej).

System ma ułatwić użytkownikowi szybkie porównanie potencjału turystycznego województw. Dzięki integracji z interfejsem Gradio i mapami Plotly, wyniki prezentowane są w sposób intuicyjny, eliminując konieczność ręcznej obróbki danych statystycznych.

---

## 2. Prawa autorskie
**a. Autorzy:** Paulina Koniecko, Alicja Skrzyńska

**b. Warunki licencyjne:** Oprogramowanie udostępniane jest na licencji **MIT**. Pozwala ona na dowolne wykorzystanie kodu pod warunkiem zachowania informacji o autorach.

---

## 3. Specyfikacja wymagań

### Moduł 1: Pobieranie i Przetwarzanie Danych
| Identyfikator | Nazwa | Opis | Priorytet | Kategoria |
| :--- | :--- | :--- | :---: | :--- |
| F1.1 | Pobieranie API | Pobieranie danych dla 11 wskaźników metodą `by-variable`. | 1 | Funkcjonalne |
| F1.2 | Normalizacja | Przeliczanie danych per capita oraz na km². | 1 | Funkcjonalne |
| NF1.1 | Obsługa błędów | Mechanizm Retry Logic dla błędów sieciowych i limitów API (429). | 1 | Pozafunkcjonalne |

### Moduł 2: Analiza Statystyczna i UI
| Identyfikator | Nazwa | Opis | Priorytet | Kategoria |
| :--- | :--- | :--- | :---: | :--- |
| F2.1 | Analiza PCA | Wyliczanie składowej PC1 na podstawie standaryzowanych danych. | 1 | Funkcjonalne |
| F2.2 | Mapa | Generowanie interaktywnego kartogramu Polski. | 1 | Funkcjonalne |
| F2.3 | Interfejs | Udostępnienie pola wyboru roku danych w GUI Gradio. | 2 | Funkcjonalne |

---

## 4. Architektura systemu/oprogramowania

**a. Architektura rozwoju:**
| Nazwa technologii | Przeznaczenie | Numer wersji |
| :--- | :--- | :--- |
| Python | Główny język programowania | 3.12 |
| Visual Studio Code | Środowisko programistyczne (IDE) | Najnowsza |
| GitHub | System kontroli wersji i repozytorium | - |
| Anaconda/Pip | Zarządzanie bibliotekami i środowiskiem | Najnowsza |

**b. Architektura uruchomieniowa:**
| Nazwa | Przeznaczenie | Numer wersji |
| :--- | :--- | :--- |
| Pandas | Manipulacja ramkami danych | 2.2.2 |
| Scikit-learn | Obliczenia PCA i standaryzacja | 1.5.1 |
| Plotly | Generowanie interaktywnych map | 5.24.1 |
| Gradio | Serwer interfejsu webowego | 6.3.0 |
| Requests | Obsługa zapytań HTTP do API GUS | 2.32.3 |

**c. Prezentacja omawiająca wykorzystywane technologie:** Pełna prezentacja merytoryczna omawiająca wskaźniki kompozytowe oraz technologie znajduje się w pliku:  
[Wskaźniki-kompozytowe_PK_AS.pdf](./Wskaźniki-kompozytowe_PK_%20AS.pdf)

---

## 5. Testy

**a. Scenariusze testów:**
1. **ST_01:** Uruchomienie pobierania danych dla roku 2022 - sprawdzenie kompletności tabeli (16 województw).
2. **ST_02:** Test odporności na limity API - wymuszenie błędu 429 i weryfikacja działania mechanizmu Retry.
3. **ST_03:** Test UI - weryfikacja wyświetlania dymka (tooltip) na mapie po najechaniu myszą.

**b. Sprawozdanie z wykonania scenariuszy testów:**
- **ST_01:** Wynik pozytywny. Dane pobrane pomyślnie, brak pustych rekordów.
- **ST_02:** Wynik pozytywny. Skrypt poprawnie wstrzymał pracę na 10s i ponowił zapytanie po wystąpieniu limitu.
- **ST_03:** Wynik pozytywny. Mapa renderuje się płynnie, dane PC1 są widoczne dla każdego regionu.
