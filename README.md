# Bottle Cap Inspection — Prototyp porównawczy

Prototyp porównujący **klasyczne metody przetwarzania obrazów** z **metodami ML/AI**
na zadaniu klasyfikacji stanu nakrętki butelki.

Celem tej fazy jest:

1. Zweryfikować czy nasz dataset jest wystarczający (rozkład klas, jakość etykiet).
2. Wstępnie porównać podejścia, żeby zdecydować które rozwijać w kolejnej fazie.

---

## Dataset

- **Źródło:** Roboflow — `bottle-cap.yolov8` (639 obrazów, format YOLOv8).
- **Klasy (5):** `Broken Cap`, `Broken Ring`, `Good Cap`, `Loose Cap`, `No Cap`.
- **Etykiety:** znormalizowane bounding boxy (`class cx cy w h`).
- Repo zawiera tylko etykiety — obrazy pobiera `data/download_dataset.py` (lub kładziesz je ręcznie w `bottle-cap.yolov8/train/images/`).

## Założenie prototypu

Zamiast pełnego pipeline'u **detekcja + klasyfikacja**, wycinamy pojedyncze nakrętki
z ground-truth bounding boxów i porównujemy klasyfikatory na tych samych crop'ach.
Dzięki temu porównanie klasy ↔ ML jest uczciwe (brak propagacji błędów z detekcji).

## Porównanie raw vs augmented

Każdą metodę (klasyczną i ML) uruchamiamy **dwa razy**:

- **raw** — trening tylko na oryginalnych crop'ach z GT bbox.
- **aug** — trening na raw + 2 dodatkowe augmentowane kopie każdego obrazu
  (blur, noise, rotate, brightness, horizontal flip).

Dzięki temu ablacja pokazuje, **ile daje augmentacja** dla każdej metody —
to pomaga zdecydować, które podejścia warto rozwijać w kolejnej fazie.

---

## Struktura projektu

```
TWM_project/
├── README.md                  <- ten plik
├── PLAN.md                    <- plan projektu (do dyskusji zespołowej)
├── requirements.txt
├── config.py                  <- wspólne ścieżki, seed, hiperparametry
├── run_all.py                 <- master script uruchamiający cały pipeline
│
├── bottle-cap.yolov8/         <- oryginalny dataset (labels + data.yaml)
│
├── data/                      <- [KOLEGA B] przygotowanie danych
│   ├── download_dataset.py    <- pobranie obrazów (Roboflow API lub info jak ręcznie)
│   ├── dataset_loader.py      <- wczytanie + crop kapsli z GT bbox
│   ├── splitter.py            <- stratified split 70/15/15
│   ├── augmentation.py        <- Albumentations: blur, noise, rotate, brightness
│   ├── preprocessing.py       <- resize + normalize
│   └── eda.py                 <- rozkład klas, przykłady, statystyki
│
├── classical/                 <- [KOLEGA A] metody klasyczne
│   ├── base_classifier.py     <- wspólny interfejs BaseClassifier
│   ├── hog_svm.py             <- HOG + SVM (implementacja referencyjna)
│   ├── edge_contour.py        <- Canny + kontury + cechy (STUB - do uzupełnienia)
│   ├── threshold_morphology.py<- Otsu + morfologia + cechy (STUB - do uzupełnienia)
│   └── run_classical.py       <- trenuje i zapisuje wyniki wszystkich metod
│
├── ml/                        <- [TY] metody ML/AI
│   ├── base_model.py          <- wspólny interfejs BaseModel
│   ├── transfer_learning.py   <- ResNet18 / MobileNetV2 fine-tuning
│   ├── feature_ml.py          <- CNN features + XGBoost / Random Forest
│   └── run_ml.py              <- trenuje i zapisuje wyniki wszystkich modeli
│
├── evaluation/                <- [WSPÓLNE] framework porównawczy
│   ├── metrics.py             <- Accuracy, F1 macro, per-class metrics, inference time
│   ├── evaluator.py           <- ewaluacja dowolnego modelu (wspólny interfejs)
│   ├── robustness.py          <- test na zaburzonych danych (blur, noise)
│   └── compare.py             <- zebranie wyników → CSV + wykresy
│
└── results/                   <- auto-generowane wyniki
    ├── metrics/               <- JSON + CSV z metrykami
    ├── plots/                 <- confusion matrix, wykresy porównawcze
    └── models/                <- zapisane modele (.pkl, .pt)
```

---

## Podział pracy w zespole

| Moduł | Odpowiedzialny | Zakres |
|---|---|---|
| `data/` | **Kolega B** | Pobranie, EDA, podział, augmentacja, preprocessing |
| `classical/` | **Kolega A** | HOG+SVM, krawędzie+kontury, Otsu+morfologia |
| `ml/` | **Ty (Adam)** | Transfer Learning, CNN features + XGBoost / RF |
| `evaluation/` + `run_all.py` | wspólnie | Framework porównawczy |

**Zasada spójności:** każda metoda (klasyczna / ML) implementuje ten sam interfejs
(`fit`, `predict`, `predict_proba`, `save`, `load`). Dzięki temu `evaluation/evaluator.py`
obsługuje każdą metodę identycznie i każdy moduł można rozwijać niezależnie.

Wspólny interfejs: zobacz `classical/base_classifier.py` i `ml/base_model.py`.

---

## Instalacja

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux / macOS
pip install -r requirements.txt
```

**Wymagany Python:** 3.10+. PyTorch instaluje się automatycznie w wersji CPU —
jeśli masz GPU, zainstaluj wariant CUDA ze strony <https://pytorch.org>.

---

## Jak uruchomić (krok po kroku)

### 1. Obrazy datasetu

Obrazy nie są w repo. Masz dwie opcje:

**A) Roboflow API** — ustaw klucz API i pobierz automatycznie:

```bash
setx ROBOFLOW_API_KEY "twoj-klucz"    # Windows, nowy terminal po tym
python data/download_dataset.py
```

**B) Ręcznie** — wrzuć obrazy `.jpg` do `bottle-cap.yolov8/train/images/`
(nazwy muszą się zgadzać z plikami `.txt` w `train/labels/`).

### 2. EDA — sprawdzenie danych

```bash
python data/eda.py
```

Wyświetli rozkład klas, statystyki bboxów, przykładowe obrazy.
Wyniki lądują w `results/plots/eda_*.png`.

### 3. Crop + podział train/val/test

```bash
python data/splitter.py
```

Tworzy `data/processed/crops/{train,val,test}/<class>/<id>.png` (stratified 70/15/15).

### 4. Uruchomienie pełnego porównania

```bash
python run_all.py                        # raw + aug (domyślnie)
python run_all.py --augmentation raw     # tylko raw (szybciej)
python run_all.py --augmentation aug     # tylko aug
```

Ten jeden skrypt:

1. Wczytuje crop'y (jeśli nie ma — wywoła splitter).
2. Uruchamia wszystkie metody klasyczne (raw i/lub aug).
3. Uruchamia wszystkie metody ML (raw i/lub aug).
4. Robi ewaluację + test odporności każdego modelu.
5. Generuje wykresy porównawcze i tabelę podsumowującą.

### 5. Alternatywnie — tylko wybrana grupa metod

```bash
python classical/run_classical.py --augmentation both    # tylko metody klasyczne
python ml/run_ml.py --augmentation both                  # tylko metody ML
python evaluation/compare.py                             # tylko agregacja wyników
```

### 6. Inferencja na pojedynczym obrazie

```bash
# lista wytrenowanych modeli
python -m evaluation.infer --list

# predykcja na jednym obrazie
python -m evaluation.infer --model transfer_resnet18_aug --image path/to/cap.jpg

# predykcja na całym folderze
python -m evaluation.infer --model hog_svm_raw --folder path/to/folder
```

---

## Gdzie szukać wyników

Każdy *run* ma nazwę `<method>_<trained_on>`, np. `hog_svm_raw`, `transfer_resnet18_aug`.

- `results/metrics/summary.csv` — tabela porównawcza **wszystkich runów** (raw + aug).
- `results/metrics/<run>.json` — szczegółowe metryki per run.
- `results/metrics/<run>_robustness.json` — odporność per run (blur, noise, brightness).
- `results/plots/confusion_<run>.png` — macierz pomyłek per run.
- `results/plots/comparison_accuracy.png` — wykres Accuracy + F1 macro, grupowany raw vs aug.
- `results/plots/comparison_speed.png` — porównanie czasu inferencji (uśredniony po raw/aug).
- `results/plots/augmentation_gain.png` — Δ(aug − raw) per metoda.
- `results/plots/robustness.png` — odporność na blur / noise.
- `results/models/<run>.{pkl,pt}` — zapisane wagi modeli.
- `results/models/<run>.meta.json` — metadane: data treningu, hiperparametry, finalne metryki.

---

## Dodawanie nowej metody (dla kolegów)

Wystarczy stworzyć klasę dziedziczącą z `BaseClassifier` (klasyczne) lub
`BaseModel` (ML), zaimplementować 5 metod i dodać instancję do odpowiedniej listy
w `classical/run_classical.py` lub `ml/run_ml.py`. Framework ewaluacyjny
automatycznie ją obsłuży.

Zobacz `classical/hog_svm.py` jako gotowy wzór dla metod klasycznych.
Zobacz `ml/transfer_learning.py` jako wzór dla sieci neuronowych.

---

## Parametry do strojenia

Wszystkie wspólne parametry są w `config.py`:

- `IMAGE_SIZE` — rozmiar wyciętego obrazu (domyślnie 128 px).
- `BBOX_PADDING` — ile kontekstu dodać wokół GT bbox (domyślnie 5%).
- `SEED` — ziarno generatora losowego.
- `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO` — proporcje splitu.
- `CNN_EPOCHS`, `CNN_LR`, `BATCH_SIZE` — hiperparametry treningu CNN.

---

## Co poza zakresem prototypu

- Pełny pipeline detekcji (YOLOv8 fine-tuning na całym obrazie).
- Demo aplikacyjne (Streamlit / Gradio).
- Szeroka optymalizacja hiperparametrów.
- Ensemble metod.

Te rzeczy dołożymy w kolejnej fazie, jeśli wyniki prototypu uznamy za zachęcające.
