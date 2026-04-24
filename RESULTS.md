# Wyniki eksperymentów — Bottle Cap Inspection

Dokument generowany ręcznie na podstawie plików w `results/` po pełnym przebiegu `run_all.py --augmentation both`. Wszystkie metryki liczone są na **test secie** (`n_test = 103`), nigdy nie tkniętym podczas treningu/walidacji.

---

## 1. Dataset (po pre-processingu)

| Podział | Liczba kropek |
|---|---|
| Train | 478 |
| Val   | 102 |
| Test  | 103 |

Po wyciągnięciu kropek z bboxów YOLO mamy łącznie **683 próbek** w 5 klasach:

| Klasa | Udział w pełnym zbiorze |
|---|---|
| Good Cap | 227 (33.2%) |
| No Cap | 162 (23.7%) |
| Loose Cap | 159 (23.3%) |
| Broken Cap | 69 (10.1%) |
| Broken Ring | 66 (9.7%) |

> Zbiór jest lekko niezbalansowany — dwie klasy „uszkodzenia" (Broken Cap, Broken Ring) są najrzadsze, dlatego F1-macro jest bardziej miarodajne niż sama accuracy.

Wykresy EDA: `results/plots/eda_class_distribution.png`, `eda_bbox_sizes.png`, `eda_examples.png`.

---

## 2. Przetestowane metody

Każdą metodę trenowano w dwóch wariantach: na surowym zbiorze (`raw`, 478 próbek) oraz na zbiorze augmentowanym offline (`aug`, 1434 próbki = 3× kopie z rotacją, flipem, zmianą jasności/kontrastu, rozmyciem i szumem gaussa).

| Kategoria | Metoda | Idea |
|---|---|---|
| Klasyczna | `hog_svm` | HOG (7056 cech) → LinearSVC z kalibracją Platta |
| Klasyczna | `edge_contour_rf` | Canny + ekstrakcja cech geometrycznych konturu → RandomForest |
| Klasyczna | `threshold_morphology_rf` | Otsu + morfologia + deskryptory kształtu → RandomForest |
| ML (hybryda) | `cnnfeat_xgboost` | Zamrożony ResNet18 jako ekstraktor (512 cech) → XGBoost |
| ML (hybryda) | `cnnfeat_random_forest` | Jak wyżej, ale klasyfikator RF |
| Deep | `transfer_resnet18` | Fine-tuning ResNet18 pretrenowany na ImageNet |
| Deep | `transfer_mobilenet_v2` | Fine-tuning MobileNetV2 |

---

## 3. Wyniki na czystym teście

### 3.1 Tabela zbiorcza

| Metoda | Wariant | Accuracy | F1-macro | Precision | Recall | Train (s) | Infer (ms/img) |
|---|---|---:|---:|---:|---:|---:|---:|
| **transfer_resnet18** | **raw** | **0.9806** | **0.9730** | 0.9863 | 0.9636 | 123.8 | 17.9 |
| cnnfeat_xgboost | raw | 0.9709 | 0.9586 | 0.9809 | 0.9455 | 13.1 | 19.5 |
| cnnfeat_xgboost | aug | 0.9709 | 0.9586 | 0.9809 | 0.9455 | 35.3 | 19.1 |
| cnnfeat_random_forest | raw | 0.9709 | 0.9586 | 0.9809 | 0.9455 | 9.1 | 19.2 |
| transfer_resnet18 | aug | 0.9515 | 0.9319 | 0.9503 | 0.9214 | 345.8 | 19.3 |
| transfer_mobilenet_v2 | aug | 0.9515 | 0.9260 | 0.9709 | 0.9091 | 251.6 | 12.3 |
| **hog_svm** | **raw** | **0.9515** | **0.9260** | 0.9709 | 0.9091 | **3.2** | **4.2** |
| hog_svm | aug | 0.9417 | 0.9059 | 0.9636 | 0.8909 | 39.9 | 4.0 |
| cnnfeat_random_forest | aug | 0.9417 | 0.9072 | 0.9664 | 0.8909 | 25.7 | 18.8 |
| threshold_morphology_rf | aug | 0.8835 | 0.8139 | 0.8437 | 0.8163 | 0.8 | 0.85 |
| threshold_morphology_rf | raw | 0.8738 | 0.7972 | 0.8173 | 0.7963 | **0.4** | **0.73** |
| edge_contour_rf | raw | 0.8155 | 0.7210 | 0.7831 | 0.7094 | 0.4 | 0.92 |
| edge_contour_rf | aug | 0.7864 | 0.6897 | 0.7538 | 0.6770 | 0.99 | 0.92 |

Wykres: `results/plots/comparison_accuracy.png`, `comparison_speed.png`.

### 3.2 Kluczowe obserwacje

1. **Zwycięzca: `transfer_resnet18_raw` (98.06% acc, 97.3% F1)** — tylko 2 błędy na 103 próbkach (jedna Broken Cap → Good Cap, jedna Broken Cap → No Cap).
2. **Najlepsza metoda klasyczna: `hog_svm_raw` (95.15% acc)** — tylko 1 pp. gorsza niż `transfer_resnet18` i 30× szybsza w treningu (3s vs 124s).
3. **Najszybsze inference:** `threshold_morphology_rf` (0.73 ms/img) i `edge_contour_rf` (0.92 ms/img) — ~20× szybciej niż modele deep.
4. **Sweet spot dla aplikacji przemysłowej:** `cnnfeat_xgboost_raw` — F1 97.1%, trening 13s, inference 19ms — dużo taniej od fine-tuningu przy takiej samej jakości.

---

## 4. Ablation study: augmentacja pomaga czy nie?

| Metoda | Acc (raw) | Acc (aug) | Δ Acc | F1 (raw) | F1 (aug) | Δ F1 |
|---|---:|---:|---:|---:|---:|---:|
| hog_svm | 0.951 | 0.942 | **−0.010** | 0.926 | 0.906 | −0.020 |
| edge_contour_rf | 0.816 | 0.786 | **−0.029** | 0.721 | 0.690 | −0.031 |
| threshold_morphology_rf | 0.874 | 0.884 | **+0.010** | 0.797 | 0.814 | +0.017 |
| cnnfeat_xgboost | 0.971 | 0.971 | 0.000 | 0.959 | 0.959 | 0.000 |
| cnnfeat_random_forest | 0.971 | 0.942 | **−0.029** | 0.959 | 0.907 | −0.051 |
| transfer_resnet18 | 0.981 | 0.951 | **−0.029** | 0.973 | 0.932 | −0.041 |

Wykres: `results/plots/augmentation_gain.png`.

### 4.1 Zaskakujący wniosek

Augmentacja **pogarsza** wyniki na czystym teście dla większości metod — z jednym wyjątkiem `threshold_morphology_rf` (tu szum pomaga, bo wymusza odporność deskryptorów kształtu).

**Dlaczego tak jest:**
- Zbiór treningowy i testowy pochodzą z tej samej, „studyjnej" dystrybucji (Roboflow) — obrazy są czyste, dobrze oświetlone, podobne kąty. Augmentacja oddala trening od tej dystrybucji.
- Przy 478 próbkach dodanie 3× kopii z losowymi zaburzeniami rozcieńcza sygnał — zamiast nauczyć się cech dyskryminacyjnych, model uczy się też inwariancji, których test nie wymaga.
- Dla CNN (transfer learning) efekt jest szczególnie widoczny, bo backbone i tak już zawiera solidne reprezentacje — dodatkowa regularyzacja augmentacją tylko szkodzi.

**Ale** — patrz sekcja 5 — augmentacja diametralnie poprawia **odporność** na zaburzenia w produkcji, więc nie można jej oceniać tylko po accuracy na czystym teście.

---

## 5. Odporność na zaburzenia (robustness)

Każdą metodę testowaliśmy dodatkowo na tym samym test secie, ale pod kontrolowanymi korupcjami (blur gaussowski, blur ruchowy, szum gaussowski, przyciemnienie, rozjaśnienie). Pełne dane: `results/metrics/<run>_robustness.json`, wykres: `results/plots/robustness.png`.

### 5.1 Accuracy pod szumem gaussowskim (najtrudniejsza korupcja)

| Metoda | Clean | Noise | Δ |
|---|---:|---:|---:|
| threshold_morphology_rf_raw | 0.874 | **0.893** | **+0.019** |
| transfer_mobilenet_v2_aug | 0.951 | 0.913 | −0.038 |
| hog_svm_aug | 0.942 | 0.874 | −0.068 |
| transfer_resnet18_raw | 0.981 | 0.650 | **−0.331** |
| hog_svm_raw | 0.951 | 0.524 | −0.427 |
| cnnfeat_random_forest_raw | 0.971 | 0.456 | **−0.515** |
| edge_contour_rf_raw | 0.816 | 0.398 | −0.418 |

### 5.2 Wnioski z odporności

1. **Augmentacja dramatycznie poprawia odporność** — `hog_svm_aug` traci 7 pp na szumie vs 43 pp dla `hog_svm_raw`. Podobnie MobileNet_aug (−4 pp) vs ResNet_raw (−33 pp).
2. **`threshold_morphology_rf` jest najbardziej odporne** — bo operuje na wysokopoziomowych deskryptorach kształtu (solidność, powierzchnia, ekscentryczność), które są niezależne od tekstury.
3. **Deep features bez augmentacji są kruche** — `cnnfeat_random_forest_raw` traci >50 pp na szumie. To pokazuje, że cechy ResNet18 (wytrenowane na ImageNet) są wrażliwe na degradację piksela.
4. **Praktyczna rekomendacja:** model produkcyjny **powinien** być trenowany z augmentacją — lepsze 95% na czystych + 91% na zaszumionych niż 98% na czystych + 65% na zaszumionych.

---

## 6. Analiza per-klasa (best model: `transfer_resnet18_raw`)

| Klasa | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Broken Cap | 1.000 | 0.818 | 0.900 | 11 |
| Broken Ring | 1.000 | 1.000 | 1.000 | 10 |
| Good Cap | 0.971 | 1.000 | 0.986 | 34 |
| Loose Cap | 1.000 | 1.000 | 1.000 | 24 |
| No Cap | 0.960 | 1.000 | 0.980 | 24 |

### 6.1 Najtrudniejsza klasa: `Broken Cap`

Wszystkie modele mylą `Broken Cap` z `Good Cap` (rzadziej z `No Cap`). Powody:
- Najmniej próbek (69 w całym zbiorze, 11 na teście).
- Wizualnie blisko „Good Cap" — pęknięcie potrafi zajmować <5% powierzchni kapsla.
- `edge_contour_rf` ma tu tylko **27% recall**, a `threshold_morphology_rf` — zaledwie **18%** (bo obie metody patrzą na globalny kształt, nie na lokalną teksturę).

### 6.2 Najłatwiejsza klasa: `No Cap` / `Loose Cap`

Praktycznie każda metoda osiąga 100% recall — to klasy o wyraźnie odmiennym globalnym wyglądzie (brak/wystawanie kapsla dają silny sygnał kształtu).

Wykresy macierzy pomyłek: `results/plots/confusion_<metoda>_<wariant>.png`.

---

## 7. Czas treningu i inferencji

| Warstwa | Najszybsze trenowanie | Najszybsze inference |
|---|---|---|
| Klasyczne | `threshold_morphology_rf_raw` (0.4s) | `threshold_morphology_rf_raw` (0.73ms) |
| ML hybryda | `cnnfeat_random_forest_raw` (9s) | `transfer_mobilenet_v2_aug` (12.3ms) |
| Deep | `transfer_resnet18_raw` (124s) | `transfer_mobilenet_v2_aug` (12.3ms) |

**HOG+SVM ma wyjątkowo korzystny stosunek jakość/prędkość:** 95% acc przy 3-sekundowym treningu i 4ms inference — idealne do szybkiej iteracji podczas rozwijania projektu.

---

## 8. Rekomendacje dla wdrożenia

| Scenariusz | Rekomendowana metoda |
|---|---|
| **Najwyższa jakość, nieograniczony compute** | `transfer_resnet18_raw` (98% acc) |
| **Produkcja z szumem/zmianami oświetlenia** | `transfer_mobilenet_v2_aug` lub `hog_svm_aug` (lepsza robustness) |
| **Ekstremalnie tani embedded (<5ms, MCU)** | `threshold_morphology_rf_raw` (88% acc, 0.7ms) |
| **Szybka iteracja / prototyp** | `hog_svm_raw` (95% acc, 3s trening) |
| **Kompromis jakość-prostota** | `cnnfeat_xgboost_raw` (97% acc, 13s trening bez GPU-fine-tuningu) |

---

## 9. Co można jeszcze ulepszyć

1. **Klasa Broken Cap** — należałoby:
   - Zebrać więcej przykładów (obecnie 69 na 683 → tylko 10%).
   - Spróbować class-weight / focal loss dla deep modeli.
   - Ewentualnie augmentacji dedykowanej tej klasie (cut-mix z good capami).
2. **Detekcja end-to-end** — aktualnie ograniczamy się do klasyfikacji crops z GT bboxów. Docelowo trzeba dodać detektor (np. YOLOv8) lub klasyczne localize-then-classify.
3. **Kalibracja progów decyzyjnych** — w kontroli jakości fałszywy alarm (Good → Broken) kosztuje mniej niż pominięcie wady (Broken → Good), więc próg sigmy należy przesunąć na rzecz recall klasy wadowej.
4. **Rozszerzony ablation augmentacji** — sprawdzić konkretnie, która transformacja (rotacja vs noise vs blur) najbardziej pomaga robustności, a najmniej szkodzi czystej accuracy.
5. **Silniejsze backbone'y** — ConvNeXt-Tiny, EfficientNet-B0 mogą podbić wynik o 1–2 pp.

---

## 10. Pliki wynikowe

```
results/
├── metrics/
│   ├── summary.csv                          # tabela zbiorcza wszystkich runów
│   ├── <run_name>.json                      # pełny raport per-klasa + CM
│   └── <run_name>_robustness.json           # accuracy pod korupcjami
├── models/
│   ├── <run_name>.{pkl,pt}                  # wytrenowany model
│   └── <run_name>.meta.json                 # provenance (hyperparamy, czas, metryki)
└── plots/
    ├── eda_*.png                            # analiza zbioru danych
    ├── comparison_accuracy.png              # porównanie głównych metryk
    ├── comparison_speed.png                 # inferencja vs accuracy
    ├── augmentation_gain.png                # Δ (aug − raw) dla każdej metody
    ├── robustness.png                       # accuracy pod korupcjami
    └── confusion_<run_name>.png             # macierze pomyłek
```

Wszystkie metryki są odtwarzalne (`SEED = 42`); pełny run od zera: `python run_all.py`.
