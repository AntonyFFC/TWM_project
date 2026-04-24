# Bottle Cap Inspection — Plan prototypu

## Kontekst

- Dataset: `bottle-cap.yolov8` (Roboflow export), 639 etykiet YOLOv8, 5 klas: `Broken Cap`, `Broken Ring`, `Good Cap`, `Loose Cap`, `No Cap`.
- Obrazy są obecnie nieobecne w repo — tylko `bottle-cap.yolov8/train/labels/*.txt` (639 plików) + `data.yaml` + `README.roboflow.txt`.
- Brak podziału val/test — zrobimy go my (stratyfikowany 70/15/15).
- Zespół 3-osobowy: Adam (ML/AI), Kolega A (metody klasyczne), Kolega B (obróbka danych).

## Uproszczenie dla prototypu

Zamiast pełnego pipeline'u detekcji, **wycinamy kapsle z ground-truth bounding boxów** i porównujemy klasyfikatory na tych samych crop'ach. To daje uczciwe porównanie metod bez efektu propagacji błędów z detekcji. W przyszłej fazie można dołożyć detektor (YOLOv8) na górę.

## Architektura (data flow)

```
bottle-cap.yolov8 (labels + images)
          |
          v
data/dataset_loader.py  -- crop z GT bbox
          |
          v
data/splitter.py        -- stratified 70/15/15
          |
   +------+------+
   |      |      |
 train   val   test
   |      |      |
   +---+--+      |
       |         |
       v         |
 data/augmentation.py
       |         |
       v         v
  [classical/]  [ml/]
       |         |
       +----+----+
            v
  evaluation/evaluator.py
            |
            v
  evaluation/compare.py
            |
            v
   results/ (metrics + plots + models)
```

## Struktura projektu

```
TWM_project/
├── README.md, PLAN.md, requirements.txt, config.py, run_all.py
├── bottle-cap.yolov8/     -- oryginalny dataset
├── data/                  -- [KOLEGA B]
├── classical/             -- [KOLEGA A]
├── ml/                    -- [TY]
├── evaluation/            -- [WSPÓLNE]
└── results/               -- [AUTO-GENEROWANE]
    ├── metrics/
    ├── plots/
    └── models/
```

## Wspólny interfejs

Każda metoda — klasyczna i ML — implementuje ten sam interfejs. Dzięki temu
`evaluation/evaluator.py` obsługuje wszystko identycznie.

```python
class BaseModel:
    name: str
    def fit(self, X_train, y_train, X_val=None, y_val=None): ...
    def predict(self, X): ...          # etykiety
    def predict_proba(self, X): ...    # prawdopodobieństwa
    def save(self, path): ...
    def load(self, path): ...
```

## Metryki porównawcze

- **Accuracy** — ogólna dokładność na teście.
- **F1 macro** — uśredniony F1 po klasach (ważny przy niezbalansowanych klasach).
- **Precision / Recall per klasa** — które klasy sprawiają problemy.
- **Confusion Matrix** — wizualizacja pomyłek.
- **Inference time** — średni czas [ms] na obraz.
- **Robustness score** — Accuracy na danych z blur + noise (dodatkowy test).

## Etapy implementacji

### Faza 1 — Dane (Kolega B)

- [ ] `data/download_dataset.py` — pobranie obrazów z Roboflow API.
- [ ] `data/eda.py` — rozkład klas, przykłady, statystyki bboxów.
- [ ] `data/splitter.py` — stratified 70/15/15, crop z GT bboxów.
- [ ] `data/augmentation.py` — Albumentations: blur, noise, rotate, brightness.
- [ ] `data/preprocessing.py` — resize, normalize.

### Faza 2 — Metody klasyczne (Kolega A)

- [ ] `classical/base_classifier.py` — (gotowe, nie zmieniać).
- [ ] `classical/hog_svm.py` — (gotowe jako wzór).
- [ ] `classical/edge_contour.py` — implementacja Canny + kontury.
- [ ] `classical/threshold_morphology.py` — Otsu + morfologia + cechy.

### Faza 3 — Metody ML (Adam)

- [ ] `ml/base_model.py` — (gotowe, nie zmieniać).
- [ ] `ml/transfer_learning.py` — ResNet18 + MobileNetV2 fine-tuning.
- [ ] `ml/feature_ml.py` — CNN features + XGBoost + Random Forest.

### Faza 4 — Ewaluacja (wspólnie)

- [ ] `evaluation/metrics.py` — metryki + timing.
- [ ] `evaluation/evaluator.py` — ewaluacja dowolnego modelu.
- [ ] `evaluation/robustness.py` — test na blur + noise.
- [ ] `evaluation/compare.py` — tabela CSV + wykresy porównawcze.

### Faza 5 (opcjonalna, poza prototypem)

- YOLOv8 fine-tuning na całym obrazie (detekcja + klasyfikacja end-to-end).
- Streamlit demo.
- Rozszerzona optymalizacja hiperparametrów.

## Podział pracy

- `data/` — **Kolega B** (przygotowanie, split, augmentacja, EDA).
- `classical/` — **Kolega A** (HOG+SVM wzór gotowy, pozostałe do uzupełnienia).
- `ml/` — **Adam** (pełna implementacja).
- `evaluation/` + `run_all.py` — **wspólnie**, szkielet gotowy.

## Wyniki

Wszystkie wyniki lądują w `results/`:

- `results/metrics/summary.csv` — tabela porównawcza.
- `results/metrics/<method>.json` — szczegółowe metryki per metoda.
- `results/plots/confusion_<method>.png` — confusion matrix.
- `results/plots/comparison_*.png` — wykresy porównawcze.
- `results/models/<method>.{pkl,pt}` — zapisane modele.
