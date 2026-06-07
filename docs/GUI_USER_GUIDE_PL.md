# Projekt TWM — Instrukcja obsługi GUI dla projektu **Bottle Cap Classification**

Aplikacja umożliwia przegląd danych, konfigurację modyfikacji danych, trenowanie modeli uczenia maszynowego (HOG+SVM oraz ResNet18), klasyfikacje opartą na klasycznych metodach przetwarzania obrazów, porównywanie wyników różnych metod oraz eksport rezultatów.

**Uruchomienie GUI:**

```bash
python run_gui.py
```

---

# Spis treści

1. Przegląd projektu
2. Architektura systemu
3. Zbiór danych
4. Zalecany workflow
5. Pasek menu
6. Zakładki

   * Data Loader
   * Augmentation Config
   * Augmentation Viewer
   * Training Pipeline
   * Machine Learning
   * Rule-Based Config & Analyze
   * Rule-Based Evaluation
   * Results Comparison
   * Export
7. Klasy klasyfikacji
8. Metryki ewaluacyjne
9. Katalogi wyjściowe

---

# Przegląd projektu

Projekt służy do klasyfikacji stanu korków butelek przy użyciu trzech niezależnych podejść:

| Rodzina metod  | Typ                            | Opis                                                                             |
| -------------- | ------------------------------ | -------------------------------------------------------------------------------- |
| **HOG + SVM**  | Feature-based Machine Learning | Klasyfikacja na podstawie ręcznie projektowanych cech HOG oraz klasyfikatora SVM |
| **ResNet18**   | Deep Learning                  | Transfer learning wykorzystujący sieć ResNet18                                   |
| **Klaszyczna** | Filtracja i przetwarzanie obrazu     | Segmentacja i heurystyki geometryczne działające na pełnych obrazach             |

Interfejs GUI został zorganizowany jako liniowy pipeline składający się z dziewięciu zakładek.

| # | Zakładka                    | Przeznaczenie                         |
| - | --------------------------- | ------------------------------------- |
| 1 | Data Loader                 | Wczytywanie i podgląd obrazów         |
| 2 | Augmentation Config         | Konfiguracja modyfikacji              |
| 3 | Augmentation Viewer         | Wizualizacja modyfikacji              |
| 4 | Training Pipeline           | Przygotowanie danych i trening modeli |
| 5 | Machine Learning            | Zarządzanie eksperymentami ML         |
| 6 | Rule-Based Config & Analyze | Konfiguracja metody klasycznej        |
| 7 | Rule-Based Evaluation       | Ewaluacja metodą klasyczna            |
| 8 | Results Comparison          | Porównanie wszystkich metod           |
| 9 | Export                      | Eksport wyników i klasyfikacja        |

---

# Architektura systemu

Projekt wykorzystuje wspólny zbiór danych, który jest przetwarzany trzema różnymi ścieżkami.

```text
Zbiór danych (YOLO)
      │
      │
      ├────────► Przycięcie ─────► HOG + SVM ──────┐
      │                                            │
      ├────────► Przycięcie ─────► ResNet18 ───────┤
      │                                            │
      └────────► Cały obraz ─► Metoda klasyczna ───┤
                                                   │
                                                   ▼
                                         Porównanie wyników
```

## HOG + SVM

1. Obraz korka jest wycinany na podstawie bounding boxa.
2. Wycinek jest skalowany do ustalonego rozmiaru.
3. Wyznaczane są deskryptory HOG.
4. Klasyfikator SVM przewiduje klasę korka.

## ResNet18

1. Obraz korka jest wycinany na podstawie bounding boxa.
2. Wycinek jest skalowany do rozmiaru wejściowego sieci.
3. Wykorzystywany jest transfer learning.
4. Sieć przewiduje prawdopodobieństwa wszystkich klas.

## Classical2

1. Analizowany jest pełny obraz.
2. Obraz jest konwertowany do skali szarości a balans bieli jest normalizowany
2. Wykonywana jest segmentacja oparta na kanale wartości V z przestrzeni HSV.
3. Wyznaczane są maski tła, butelki i korka.
4. Obliczane są cechy geometryczne charakterystyczne dla każdej z klas.
5. Reguły eksperckie przypisują klasę.

---

# Zbiór danych

Projekt wykorzystuje zbiór danych zapisany w strukturze zgodnej z YOLO.

Przykładowa struktura katalogów (nazwa zdjęcia przykładowa):

```text
bottle-cap.yolov8/
└── train/
    ├── images/
    │   ├── image001.jpg
    │   └── ...
    └── labels/
        ├── image001.txt
        └── ...
```

Etykiety są przechowywane w plikach tekstowych i zawierają informacje o klasie detektu a także odpowiednie współrzędne odpowiadających im pól.

---

# Zalecany workflow

## Pierwsza konfiguracja

1. Upewnij się, że zbiór danych (obrazy i etykiety) znajduje się w katalogu `bottle-cap.yolov8/train/`.
2. Otwórz zakładkę **Training Pipeline**.
3. Zweryfikuj sekcję **Prerequisites**.
4. Kliknij **Run Split**, jeśli nie istnieją jeszcze zbiory train/validation/test.
5. Opcjonalnie uruchom **Run EDA** do zwizualizowania wstępnych wykresów w `results/plots/`.

---

## Ścieżka Machine Learning

1. **Data Loader** — sprawdź poprawność wczytywania obrazów.
2. **Augmentation Config / Viewer** — skonfiguruj modyfikacje.
3. **Training Pipeline** — ustaw hiperparametry i uruchom **Run Full Pipeline**.
4. **Machine Learning** — po treningu przeanalizuj wybrane zdjęcia, macierze pomyłek oraz metryki.
5. **Results Comparison** — porównaj wszystkie modele.
6. **Export** — wygeneruj materiały demonstracyjne.

---

## Ścieżka Klasyczna

1. **Rule-Based Config & Analyze** — skonfiguruj parametry.
2. **Rule-Based Evaluation** — uruchom ewaluację na zbiorze danych.
3. **Results Comparison** — porównaj wyniki z metodami ML.

---

# Pasek menu

| Menu | Element                       | Działanie                              |
| ---- | ----------------------------- | -------------------------------------- |
| File | Open results folder           | Otwiera katalog `results/`             |
| File | Exit                          | Zamyka aplikację                       |
| View | Open rule-based errors folder | Otwiera katalog błędów met. klasycznej |
| View | Reset Layout                  | Przywraca domyślny rozmiar okna        |
| Help | About                         | Wyświetla informacje o aplikacji       |

Pasek statusu wyświetla aktualnie wykonywaną operację oraz nazwę ostatnio wczytanego obrazu.

---

# Zakładki

---

# 1. Data Loader

![Data Loader](../gui/screenshots/01-data-loader.png)

## Cel

Centralne miejsce do wyboru obrazów używanych przez pozostałe zakładki.

## Sposób użycia

1. Kliknij **Browse Raw Images**, aby wybrać obraz z:

   ```text
   bottle-cap.yolov8/train/images/
   ```

2. Lub kliknij **Browse Processed Crops**, aby wybrać crop z:

   ```text
   data/processed/crops/
   ```

3. Zweryfikuj metadane obrazu:

   * ścieżkę,
   * rozdzielczość,
   * rozmiar pliku.

4. Sprawdź podgląd obrazu.

## Uwaga

Zakładki **Augmentation Viewer**, **Rule-Based Analyze** oraz część funkcji **Export** korzystają z aktualnie wybranego obrazu poprzez opcję **Use Data Loader**.

---

# 2. Augmentation Config

![Augmentation Config](../gui/screenshots/02-augmentation-config.png)

## Cel

Konfiguracja stochastycznego pipeline'u modyfikacji obrazu wykorzystywanego podczas:

* treningu modeli ML,
* ewaluacji metod klasycznych na zmodyfikowanych danych.

## Dostępne modyfikacje

| Modyfikacja     | Zastosowanie                   |
| --------------- | ------------------------------ |
| Rotation        | Symulacja przechylenia butelki |
| Horizontal Flip | Odbicie lustrzane              |
| Brightness      | Zmiany oświetlenia             |
| Contrast        | Zmiany kontrastu               |
| Gaussian Blur   | Rozmycie obrazu                |
| Motion Blur     | Rozmycie ruchu                 |
| Gaussian Noise  | Szum czujnika                  |

Każda modyfikacja posiada:
* włącznik,
* odpowiedni zestaw parametrów,
* prawdopodobieństwo wystąpienia.

## Ustawienia wizualizacji

### Number of copies

Liczba próbek generowanych przez pipeline.

### View mode

* `individual`
* `copies`
* `both`

### Reset to Defaults

Przywraca ustawienia domyślne.

---

# 3. Augmentation Viewer

![Augmentation Viewer](../gui/screenshots/03-augmentation-viewer.png)

## Cel

Wizualna weryfikacja modyfikacji obrazu przed treningiem lub ewaluacją.

## Sposób użycia

1. Wczytaj obraz w zakładce **Data Loader**.
2. Skonfiguruj modyfikacje.
3. Kliknij **Generate Visualization**.
4. Opcjonalnie wybierz lokalizację zapisu PNG.

## Interpretacja wyników

### Górny wiersz

Każda modyfikacja zastosowana osobno do oryginalnego obrazu.

### Dolny wiersz

Wyniki losowego przejścia przez pełny pipeline modyfikacji.

Każda kopia generowana jest niezależnie od pozostałych.

---

# 4. Training Pipeline

![Training Pipeline](../gui/screenshots/04-training-pipeline.png)

## Cel

Kompletna orkiestracja procesu uczenia:

* EDA (eksploracyjna analiza danych),
* podział danych,
* trening HOG+SVM,
* trening ResNet18,
* generowanie wykresów,
* generowanie demonstracji.

---

## Prerequisites

Wyświetla:

* liczbę obrazów,
* liczbę etykiet,
* status podziału train/val/test.

Przycisk:

**Refresh status**

odświeża stan po zmianach dokonanych poza GUI.

---

## Hyperparameters

Zmiany mogą zostać zapisane do:

```python
config.py
```

lub ponownie wczytane.

### Kluczowe parametry

| Parametr         | Znaczenie                    |
| ---------------- | ---------------------------- |
| Train ratio      | Udział zbioru treningowego   |
| Validation ratio | Udział zbioru walidacyjnego  |
| Test ratio       | Udział zbioru testowego      |
| Crop size        | Rozmiar przycięcia           |
| BBox padding     | Margines wokół bounding boxa |
| Batch size       | Rozmiar batcha               |
| CNN epochs       | Liczba epok                  |
| LR               | Learning rate                |

---

## Pipeline options

### Augmentation

| Opcja | Znaczenie                      |
| ----- | ------------------------------ |
| raw   | tylko dane oryginalne          |
| aug   | tylko dane modyfikowane        |
| both  | dane oryginalne i modyfikowane |

### Skip

Pozwala pominąć wybrane kroki:

* EDA
* Split
* HOG+SVM
* Neural Network
* Demo

---

## Przyciski

### Run EDA

Generuje statystyki i wykresy.

### Run Split

Tworzy cropy oraz podział train/val/test.

### Run Full Pipeline

Uruchamia wszystkie niepominięte kroki.

---

# 5. Machine Learning

![Machine Learning](../gui/screenshots/05-machine-learning.png)

## Cel

Trening i analiza wyników dla:

* HOG + SVM (lewa strona)
* ResNet18 (prawa strona)

---

## Dla każdej kolumny

### Train on

* raw - dane oryginalne
* aug - dane zmodyfikowane
* both - dane oryginalne i zmodyfikowane

### Run training

Uruchamia trening.

### Run

Wybór zapisanego eksperymentu.

### Refresh

Odświeża listę dostępnych eksperymentów.

---

## Wyświetlane metryki

* Dokładność 
* Precyzja
* Czułość 
* F1
* Czas trenowania
* Czas analizy
* Liczba próbek

Poniżej metryk prezentowana jest macierz pomyłek.

---

# 6. Rule-Based Config & Analyze

![Rule-Based Config & Analyze](../gui/screenshots/06-rule-based-config-analyze.png)

## Cel

Konfiguracja i debugowanie algorytmu Metody Klasycznej.

---

## Lewy panel

### Preset

Wczytywanie i zapis presetów JSON.

Lokalizacja:

```text
classical/presets/
```

### Grupy parametrów

* segmentacja
* morfologia
* detekcja korka
* detekcja luźnego korka
* detekcja uszkodzonego korka
* wykrywanie obrączki korka
* wykrywanie lini pomiędzy warstwami

---

## Prawy panel

### Analiza pojedynczego obrazu

1. Ustaw katalog z etykietami danych.
2. Wybierz obraz.
3. Kliknij **Analyze**.

---

## Wynik klasyfikacji

### Expected

Dane z etykiety referencyjna ze zbioru danych.

### Predicted

Wynik uzyskany przy pomocy klasycznej metody.

### Match

* Zgodność
* Częściowa zgodność
* Niezgodność


### Flags
Flagi opisujace heurystyki, które zostały aktywowane do wykrycia odpowiedniej klasy 

## Zakładki podglądu

* Original - zdjęcie wejściowe
* Annotated - zdjęcie z komenatrzami po analizie
* Bottle Mask - warstwa butelki
* Cap Mask - warstwa korka
* Background - warstwa tła

---

# 7. Rule-Based Evaluation

![Rule-Based Evaluation](../gui/screenshots/07-rule-based-evaluation.png)

## Cel

Masowa ewaluacja metody klasycznej na oznaczonym zbiorze danych.

---

## Konfiguracja

| Element                 | Opis                           |
| ----------------------- | ------------------------------ |
| Images                  | Ścieżka do obrazów             |
| Labels                  | Ścieżka do etykiet             |
| Errors                  | Katalog błędów                 |
| Preset                  | Preset JSON                    |
| Evaluate                | oryginalne / wzmocnione / oba  |
| Copies                  | Liczba modyfikowanych kopii    |
| Use Augmentation Config | Użyj konfiguracji modyfikacji  |
| Export Errors           | Zapisz błędy                   |
| Include Partial         | Uwzględnij częściowe trafienia |

---

## Układ zakładki

### Lewa strona

* metryki
* macierz pomyłek
* log
* czas klasyfikacji

### Prawa strona

Tabela błędnych klasyfikacji.

Przycisk:

**Re-analyze**

uruchamia ponowną analizę dla wybranego przypadku.

---

## Wyniki

Po zakończeniu ewaluacji generowany jest plik:

```text
results/metrics/classical2_{preset}_{raw|aug}.json
```

oraz automatycznie aktualizowane jest porównanie wyników.

---

# 8. Results Comparison

![Results Comparison](../gui/screenshots/08-results-comparison.png)

## Cel

Wspólne porównanie wszystkich metod - HOG+SVM, ResNet18 oraz klasycznej.

---

## Obsługa

### Refresh Table

Ponownie wczytuje:

```text
results/metrics/summary.csv
```

### Run Comparison

Tworzy nowe zestawienie na podstawie raportów JSON.

---

## Kolumny tabeli

| Kolumna      | Znaczenie               |
| ------------ | ----------------------- |
| method       | Nazwa eksperymentu      |
| type         | Typ metody              |
| trained_on   | raw / aug               |
| accuracy     | Accuracy                |
| f1_macro     | F1                      |
| inference_ms | Średni czas klasyfikacji|
| train_time_s | Czas treningu           |

---

## Dostępne wykresy

* comparison_accuracy.png
* comparison_speed.png
* augmentation_gain.png
* robustness.png

---

# 9. Export

![Export](../gui/screenshots/09-export.png)

## Cel

Eksport wyników, klasyfikacja pojedynczych obrazów oraz generowanie materiałów demonstracyjnych.

---

## Eksport

### Export summary.csv

Eksport tabeli wyników.

### Export results bundle (zip)

Eksport:

* modeli,
* metryk,
* wykresów.

### Open results folder

Otwiera katalog wyników.

---

## Inference - klasyfikacja wybranego zdjęcia

1. Kliknij **Refresh models**.
2. Wybierz model.
3. Wybierz obraz.
4. Kliknij **Predict**.

Wyświetlone zostaną prawdopodobieństwa wszystkich klas.

---

## Demo Visualization

1. Wybierz eksperyment HOG+SVM.
2. Wybierz eksperyment ResNet18.
3. Kliknij **Generate demo PNG**.

Plik zostanie zapisany jako:

```text
results/plots/demo_predictions.png
```

---

# Klasy klasyfikacji

| Kod | Nazwa       | Opis                |
| --- | ----------- | --------------------|
| 0   | Broken Cap  | Uszkodzony korek    |
| 1   | Broken Ring | Uszkodzony kołnierz |
| 2   | Good Cap    | Dobry korek         |
| 3   | Loose Cap   | Niedokręcony korek  |
| 4   | No Cap      | Brak korka          |

Metody ML działają na cropach wyciętych z bounding boxów.

Metoda klasyczna analizuje pełne obrazy i mapuje wynik do tych samych pięciu klas.

---

# Metryki ewaluacyjne

## Accuracy

Odsetek poprawnie sklasyfikowanych próbek.

## Precision

Jaki procent predykcji danej klasy był poprawny.

## Recall

Jaki procent rzeczywistych przykładów klasy został wykryty.

## F1 Score

Średnia harmoniczna precision i recall.

## Macro F1

Średnia F1 liczona niezależnie dla każdej klasy.

---

## Partial Match

Gdy metoda klasyczna wykrywa tylko część defektów, uwzględniane jest to jako częściowe dopasowanie, które ma wagę `0.5` przy obliczaniu metryk

---

# Katalogi wyjściowe

| Ścieżka                    | Zawartość                                       |
| -------------------------- | ----------------------------------------------- |
| `results/metrics/`         | Raporty JSON oraz `summary.csv`                 |
| `results/plots/`           | Wykresy, porównania, macierze pomyłek           |
| `results/models/`          | Modele HOG+SVM (`.pkl`) oraz ResNet18 (`.pt`)   |
| `classical/result/errors/` | Przypadki błędnej klasyfikacji                  |
| `classical/presets/`       | Presety do metody klasycznej                    |
| `data/processed/crops/`    | Zbiór przyciętych danych                        |

---

# Screenshots

Wszystkie zrzuty ekranu znajdują się w katalogu:

```text
gui/screenshots/
```

i odpowiadają kolejności zakładek opisanych w tej dokumentacji.

---

W celu uzyskania informacji dotyczących konfiguracji projektu oraz uruchamiania z linii poleceń, zapoznaj się z głównym plikiem:

```text
README.md
```
