# Bottle Cap Inspection — klasyka vs. sieci

Prosty prototyp porównujący **klasyczną metodę CV** z **siecią neuronową**
na zadaniu klasyfikacji stanu nakrętki butelki.

| Metoda | Typ | Pipeline |
|---|---|---|
| **HOG + SVM** | klasyczna | gradienty HOG → StandardScaler → LinearSVC (kalibrowany) |
| **ResNet18** | deep learning | transfer learning (ImageNet → fine-tuning ostatniego bloku) |

Porównujemy je na tych samych crop'ach z GT bounding boxów — żeby porównanie
było uczciwe i łatwe do zinterpretowania.

---

## Dataset

- **Źródło:** Roboflow `bottle-cap.yolov8` (639 zdjęć).
- **Klasy (5):** `Broken Cap`, `Broken Ring`, `Good Cap`, `Loose Cap`, `No Cap`.
- Z każdego bounding boxa wycinamy kwadratowy crop (`128 × 128 px`).
- Stratified split 70 / 15 / 15 (train / val / test).

## Augmentacja (raw vs aug)

Każdy model trenujemy **dwa razy**:

- **raw** — sam dataset, bez augmentacji,
- **aug** — dataset + 2 augmentowane kopie (rotacje, blur, szum, jasność, flip).

Dzięki temu w wykresach widać, ile augmentacja faktycznie daje każdej metodzie.

---

## Struktura projektu

```
TWM_project/
├── README.md
├── requirements.txt
├── config.py              <- ścieżki, klasy, seed, hiperparametry
├── run_all.py             <- pipeline: dane → trening → ewaluacja → demo
├── demo.py                <- ładny side-by-side wizualny PNG do prezentacji
│
├── bottle-cap.yolov8/     <- dataset (etykiety + data.yaml; obrazy pobierane)
│
├── data/                  <- przygotowanie danych
│   ├── download_dataset.py
│   ├── dataset_loader.py  <- crop kapsli z GT bbox
│   ├── splitter.py        <- stratified split
│   ├── augmentation.py
│   ├── preprocessing.py
│   └── eda.py
│
├── classical/             <- HOG + SVM
│   ├── base_classifier.py
│   ├── hog_svm.py
│   └── run_classical.py
│
├── ml/                    <- ResNet18 transfer learning
│   ├── base_model.py
│   ├── transfer_learning.py
│   └── run_ml.py
│
├── evaluation/            <- framework porównawczy
│   ├── metrics.py
│   ├── evaluator.py
│   ├── robustness.py
│   ├── compare.py
│   └── infer.py
│
└── results/               <- auto-generowane wyniki
    ├── metrics/
    ├── plots/
    └── models/
```

Obie metody implementują ten sam interfejs (`fit`, `predict`, `predict_proba`,
`save`, `load`), więc framework ewaluacyjny obsługuje je identycznie.

---

## Instalacja

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

Wymagany Python ≥ 3.10. PyTorch instaluje się w wersji CPU; jeśli masz GPU
i chcesz CUDA, doinstaluj wariant ze strony <https://pytorch.org>.

---

## Jak uruchomić

### 1. Pobierz obrazy datasetu

**Opcja A — Roboflow API:**
```bash
setx ROBOFLOW_API_KEY "twoj-klucz"   # Windows, otwórz nowy terminal
python data/download_dataset.py
```

**Opcja B — ręcznie:** wrzuć `.jpg` do `bottle-cap.yolov8/train/images/`
(nazwy muszą zgadzać się z plikami `.txt` w `train/labels/`).

### 2. Cały pipeline jednym strzałem

```bash
python run_all.py                      # raw + aug, oba modele, demo
python run_all.py --augmentation aug   # tylko aug (najszybciej)
```

To uruchamia po kolei: EDA → split → trening obu metod → ewaluacja
→ wykresy → wizualne demo PNG.

### 3. Pojedyncze etapy

```bash
python data/eda.py                     # tylko EDA
python data/splitter.py                # tylko split
python classical/run_classical.py      # tylko HOG+SVM
python ml/run_ml.py                    # tylko ResNet18
python evaluation/compare.py           # tylko agregacja wyników
python demo.py                         # tylko wizualne demo
```

### 4. Inferencja na pojedynczym obrazie

```bash
python -m evaluation.infer --list                              # lista modeli
python -m evaluation.infer --model resnet18_aug --image cap.jpg
python -m evaluation.infer --model hog_svm_aug --folder test/
```

---

## Co zobaczyć w `results/`

Każdy *run* ma nazwę `<method>_<trained_on>`, np. `hog_svm_aug`, `resnet18_raw`.

| Plik | Co tam jest |
|---|---|
| `results/metrics/summary.csv` | tabela porównawcza wszystkich runów |
| `results/metrics/<run>.json` | pełne metryki per run |
| `results/metrics/<run>_robustness.json` | odporność per run (blur, noise, ...) |
| `results/plots/comparison_accuracy.png` | bar chart: Accuracy + F1 macro, raw vs aug |
| `results/plots/comparison_speed.png` | czas inferencji (ms / obraz) |
| `results/plots/augmentation_gain.png` | Δ (aug − raw) per metoda |
| `results/plots/robustness.png` | accuracy pod różnymi zaburzeniami |
| `results/plots/confusion_<run>.png` | macierz pomyłek per run |
| `results/plots/demo_predictions.png` | **wizualne side-by-side demo** (do slajdu) |
| `results/models/<run>.{pkl,pt}` | zapisane wagi |
| `results/models/<run>.meta.json` | metadane: data, hiperparametry, finalne metryki |

---

## Najczęstsze parametry (`config.py`)

- `IMAGE_SIZE` — rozmiar wycinka (128 px).
- `BBOX_PADDING` — ile kontekstu wokół GT bbox (5%).
- `SEED` — ziarno generatora losowego.
- `CNN_EPOCHS`, `CNN_LR`, `BATCH_SIZE` — hiperparametry treningu ResNet18.

---

## Co poza zakresem

- pełny pipeline detekcji (YOLOv8 fine-tuning na całym obrazie),
- aplikacja webowa (Streamlit/Gradio),
- ensemble metod,
- szeroka optymalizacja hiperparametrów.
