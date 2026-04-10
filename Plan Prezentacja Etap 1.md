# **Prezentacja Etap 1: Założenia i projekt techniczny**

**Temat:** Kontrola jakości na linii produkcyjnej – Wizyjna detekcja poprawności zakręcenia butelek.

## **Slajd 1: Tytuł i Wprowadzenie**

* **Tytuł projektu:** Automatyczny system wizyjny do weryfikacji poprawności zamknięcia butelek.  
* **Problem biznesowy:** Wady zamknięć powodują wycieki, psucie się produktu i straty wizerunkowe firmy. Wymagana jest szybka i niezawodna weryfikacja 100% produktów schodzących z taśmy.

## **Slajd 2: Opis tematu – Cel i Założenia**

* **Cel projektu:** Opracowanie i implementacja wieloetapowego systemu komputerowego widzenia (Computer Vision), który w czasie rzeczywistym zweryfikuje poprawność nałożenia nakrętki.  
* **Rozpoznawane typy wad:**  
  * Nakrętka nałożona krzywo (nieprawidłowy kąt).  
  * Nakrętka niedokręcona (zbyt duża szczelina między nakrętką a pierścieniem zabezpieczającym).  
  * Całkowity brak nakrętki.  
* **Założenia środowiskowe (Warunki pracy):**  
  * **Prędkość:** Wysoka prędkość linii produkcyjnej. Algorytm musi charakteryzować się niskim czasem opóźnienia (niska latencja, np. \< 50 ms).  
  * **Oświetlenie:** Kontrolowane oświetlenie przemysłowe (np. *backlight*), zapewniające optymalny kontrast.

## **Slajd 3: Przegląd istniejących rozwiązań**

Analiza stanu obecnego (State of the Art) w przemyśle:

1. **Kompaktowe Kamery Smart (np. Cognex, Keyence)**  
   * *Mocne strony:* Niezawodność, graficzne interfejsy konfiguracji, wbudowane protokoły przemysłowe.  
   * *Słabe strony:* Architektura "czarnej skrzynki", brak dostępu do niskopoziomowego kodu, bardzo wysokie koszty wdrożenia przy rozproszonych systemach.  
2. **Profilometry Laserowe 3D**  
   * *Mocne strony:* Precyzyjny pomiar głębokości i skoku gwintu, odporność na zmienność kolorów nakrętek.  
   * *Słabe strony:* Wolniejsze przetwarzanie, znaczny koszt, skomplikowana kalibracja.  
3. **Nasze rozwiązanie: IPC \+ System Hybrydowy (OpenCV / AI)**  
   * *Mocne strony:* Pełna kontrola nad algorytmami i rurociągiem danych, możliwość łączenia klasycznego przetwarzania obrazu z sieciami neuronowymi (Deep Learning), niski koszt sprzętowy.

## **Slajd 4: Omówienie zbioru danych (Dataset)**

* **Źródło i struktura:** Zbiór obrazów przemysłowych pozyskany z platformy otwartej (np. Roboflow) / zbiór autorski.  
* **Podział zbioru:** standardowy podział na zbiór treningowy (do kalibracji parametrów i modeli), walidacyjny oraz testowy.  
* **Augmentacja danych (Data Augmentation):**  
  * W celu symulacji trudnych warunków przemysłowych i uodpornienia systemu, rozszerzymy zbiór danych o sztuczne zakłócenia.  
  * Przykłady: *Motion Blur* (symulacja drgań taśmociągu), *Gaussian Noise* (szum matrycy), zmienne wartości jasności (symulacja starzejącego się oświetlacza LED) oraz losowe obroty w płaszczyźnie 2D.

## **Slajd 5: Projekt techniczny rozwiązania (Architektura Systemu)**

*(Na tym slajdzie znajdzie się schemat blokowy całego rurociągu)*  
**1\. Blok Akwizycji i Wczytywania Danych:**

* **Zadanie:** Pobranie obrazu i symulacja sygnału z czujnika (Trigger).  
* **Algorytmy/Funkcje:** Funkcje wejścia/wyjścia (np. cv2.imread do ładowania z dysku, API producenta kamery).  
* **Dane wyjściowe (do Bloku 2):** Surowa macierz obrazu 2D (struktura typu *NumPy array*, format BGR lub Grayscale, 8-bit).

**2\. Blok Preprocessingu:**

* **Zadanie:** Przygotowanie obrazu, redukcja zakłóceń i wyizolowanie obszaru zainteresowania (ROI).  
* **Algorytmy:** Konwersja przestrzeni barw, filtracja dolnoprzepustowa (Rozmycie Gaussa), progowanie adaptacyjne / algorytm Otsu (Thresholding), operacje morfologiczne (Zamknięcie/Otwarcie) w celu eliminacji szumu punktowego.  
* **Dane wyjściowe (do Bloku 3):** Binarna macierz obrazu (obraz czarno-biały / Maska obiektów).

**3\. Blok Ekstrakcji Cech i Detekcji Wad:**

* **Zadanie:** Wyciągnięcie konkretnych wartości liczbowych z maski.  
* **Algorytmy CV:** Ekstrakcja konturów (findContours), wyznaczanie minimalnego prostokąta obwiedniego (minAreaRect), geometria analityczna do liczenia pikselowej odległości między elementami nakrętki a szyjką (Gap Inspection).  
* **Algorytmy ML:** Ekstrakcja cech tekstury (np. HOG) \+ klasyfikator wektorów nośnych (SVM) lub miniaturowa sieć CNN oceniająca wycięty profil.  
* **Dane wyjściowe (do Bloku 4):** Zestaw cech liczbowych (Typy *Float/Int*, np. kat \= 1.5, szczelina \= 12px, pewnosc\_ML \= 0.95).

**4\. Blok Decyzyjny i Raportowania:**

* **Zadanie:** Podjęcie ostatecznej decyzji OK/NOK na podstawie logiki biznesowej.  
* **Algorytmy:** System regułowy oparty na progach tolerancji (tzw. hard-limits, np. jeśli $k\\text{ą}t \> 2.0^\\circ \\rightarrow \\text{NOK}$), agregacja wyników z analizy klasycznej i ML.  
* **Dane wyjściowe (Wynik końcowy):** Zmienna logiczna stanu (Typ *Boolean*: PASS/FAIL), logi diagnostyczne (*JSON/String*) oraz sygnał wyjściowy przesyłany do sterownika PLC celem odrzutu braku.

## **Slajd 6: Podział prac w zespole (Organizacja Projektu)**

Ze względu na złożoność zagadnienia i potrzebę rzetelnej oceny skuteczności, projekt podzielono na trzy główne obszary badawczo-inżynierskie:

1. **Inżynier ds. Potoku Danych i Architektury (Osoba 1):**  
   * Przygotowanie zbioru danych i pipeline'u augmentacji (symulacje rozmycia, oświetlenia).  
   * Zbudowanie struktury aplikacji i ewentualnego graficznego interfejsu testowego (GUI) wyświetlającego wyniki na żywo.  
2. **Inżynier ds. Przetwarzania Obrazów \- Classic CV (Osoba 2):**  
   * Zaprojektowanie algorytmów klasycznych (OpenCV).  
   * Tworzenie logiki znajdującej kontury, mierzącej dystanse (szczeliny) i analizującej geometrię bez użycia sieci neuronowych.  
   * Optymalizacja kodu pod kątem najniższych opóźnień (zgodnie z logiką PLC/IPC).  
3. **Inżynier ds. Analizy Danych / ML (Osoba 3):**  
   * Trenowanie i walidacja niewielkiego modelu uczenia maszynowego (np. z użyciem biblioteki scikit-learn lub prostej sieci klasyfikującej).  
   * Przeprowadzenie analizy porównawczej: zestawienie wydajności (czasu wykonania) i dokładności klasycznych metod numerycznych w stosunku do metod opartych na uczeniu maszynowym.