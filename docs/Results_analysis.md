
![Export](../results/plots/comparison_accuracy.png)

Analizując wykres porównujący dokładność zastosowanych metod, można zauważyć, że w zadaniu detekcji wad korków najlepsze wyniki osiągnęła metoda oparta na głębokim uczeniu maszynowym z wykorzystaniem sieci ResNet18. Dokładność klasyfikacji wyniosła 98% dla danych niezmodyfikowanych oraz 96% dla danych poddanych augmentacji. Metoda oparta na deskryptorach HOG i klasyfikatorze SVM uzyskała nieznacznie gorsze rezultaty.

Najniższą dokładność osiągnęła metoda klasyczna. Pomimo zastosowania wielu filtrów oraz heurystyk nie udało się uzyskać wyników zbliżonych do dwóch pozostałych metod. W przypadku zmodyfikowanego zbioru danych widoczny jest bardzo duży spadek dokładności. Najprawdopodobniej wynika to z braku odporności zastosowanych heurystyk na zakłócenia wprowadzane przez operacje augmentacji, takie jak rotacja czy rozmycie obrazu.

![Export](../results/plots/confusion_resnet18_aug.png)

Analizując macierz pomyłek dla modelu ResNet18 wytrenowanego na danych zmodyfikowanych, można zauważyć, że sieć największe trudności miała z poprawną klasyfikacją uszkodzonych korków (*Broken Cap*). Analizując konkretne przypadki błędnej klasyfikacji (**TODO – wstawić przykładowe obrazy błędnie sklasyfikowane przez model**), można zauważyć, że **TODO**.

W przypadku pozostałych klas model osiągnął bardzo wysoką skuteczność, a liczba błędnych klasyfikacji była znikoma.

![Export](../results/plots/confusion_hog_svm_aug.png)

Podobnie jak w poprzednim przypadku, metoda wykorzystująca deskryptory HOG oraz klasyfikator SVM największe trudności miała z obrazami przedstawiającymi uszkodzony korek. Dla pozostałych klas osiągnięto bardzo wysoką skuteczność klasyfikacji, a liczba pomyłek była niewielka.

![Export](../results/plots/confusion_classical2_default_raw.png)

W przypadku metody klasycznej, testowanej na całym dostępnym zbiorze danych, udało się osiągnąć dokładność na poziomie około 80%. Analizując macierz pomyłek, można zauważyć, że podobnie jak w poprzednich metodach największą trudność stanowiła identyfikacja uszkodzonych korków. Dodatkowo pierwszy wiersz macierzy wskazuje, że próbki należące do klasy *Broken Cap* były często błędnie klasyfikowane jako inne rodzaje defektów.

![Export](../results/plots/confusion_classical2_default_aug.png)

Po zastosowaniu augmentacji danych wejściowych dokładność klasyfikacji spadła poniżej 50%. Podobnie jak w przypadku danych niezmodyfikowanych, największym problemem pozostała poprawna identyfikacja uszkodzonych korków, dla których liczba błędnych klasyfikacji była szczególnie wysoka. Uzyskane wyniki wskazują, że opracowana metoda bardzo słabo radzi sobie ze zmianami danych wejściowych i jest mało odporna na zakłócenia generowane przez augmentację.

![Export](../results/plots/comparison_speed.png)

Analizując średni czas potrzebny na sklasyfikowanie pojedynczego obrazu, można zauważyć, że metoda klasyczna potrzebowała około 3 ms na obraz. Wynika to z relatywnie prostej metodologii klasyfikacji opartej na filtrach i heurystykach.

Metoda wykorzystująca HOG oraz SVM potrzebowała około 10 ms na obraz. Oznacza to możliwość przetwarzania około 100 obrazów na sekundę, co stanowi bardzo dobry wynik nawet w zastosowaniach przemysłowych.

Najwięcej czasu na analizę pojedynczego obrazu wymagała sieć ResNet18. Średni czas inferencji wyniósł około 40,98 ms, co jest wartością około czterokrotnie większą niż w przypadku metody HOG+SVM. Wynika to z konieczności wykonania pełnego przejścia obrazu przez wielowarstwową sieć neuronową w celu uzyskania końcowej predykcji.

## Podsumowanie

Najlepsze wyniki klasyfikacji uzyskano przy wykorzystaniu głębokiego uczenia maszynowego. Kosztem wysokiej dokładności jest jednak większy czas potrzebny na analizę pojedynczego obrazu. W przypadku metody opartej na HOG+SVM udało się osiągnąć nieznacznie gorszą dokładność przy jednoczesnym znacznym skróceniu czasu inferencji.

Obie metody uczenia maszynowego wykazały wysoką skuteczność zarówno dla danych oryginalnych, jak i danych poddanych augmentacji, co świadczy o ich dobrej odporności na zmiany danych wejściowych.

W przypadku klasycznych metod analizy obrazu uzyskano najkrótszy czas przetwarzania, jednak kosztem najniższej dokładności klasyfikacji. Szczegółowa analiza wyników wskazuje, że opracowana metoda jest mało odporna na zmiany danych wejściowych i znacząco traci skuteczność po zastosowaniu augmentacji.

Dodatkowo przygotowanie oraz strojenie heurystyk stanowi proces czasochłonny i wymagający dużej wiedzy eksperckiej. Trudno jest również przewidzieć wszystkie możliwe warianty występowania wad korków, szczególnie te pojawiające się sporadycznie w zbiorze danych. Powoduje to, że metody klasyczne są znacznie mniej skalowalne i trudniejsze w utrzymaniu niż nowoczesne rozwiązania oparte na uczeniu maszynowym.

