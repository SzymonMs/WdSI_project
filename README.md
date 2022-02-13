# WdSI_project

Projekt na zaliczenie przedmiotu Wprowadzenie do Sztucznej inteligencji. Zadaniem programu jest klasyfikacja znaków drogowych na dwie grupy: speedlimit oznaczone jako 0 oraz other oznaczone jako 1. W projekcie nie wykonano zadania detekcji znaku, program klasyfikuje całe zdjęcia. Przyjęto następujące założenie: jeżeli na zdjęciu znajduje się kilka znaków drogowych i jest w tej grupie znak speedlimit to całe zdjęcie, zostaje oznaczone jako speedlimit (klasa 0), jeżeli nie ma na zdjęciu znaku speedlimit to obraz jest oznaczany jako ogher (klasa 1). Założenie to ma uzasadnienie w koncepcji projektu, polegającej na klasyfikacji zdjęć na podstawie wystąpienia znaku ograniczenia prędkości lub nie. Program nie wykonuje detekcji położenie znaku, więc niepotrzebne jest dzielenie obrazu na okna z różnymi znakami. W projekcie przyjęto również kryterium, aby znak zajmował 1/10 wysokości i szerokości zdjęcia, co sprowadza się do sprawdzenia, czy pole znaku zajmuje 1/100 pola całego zdjęcia.

# Dokumentacja kodu

Wszystkie potrzebne biblioteki/ pakiety oprogramowania zostały zaimportowane na górze kodu.
```python
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import csv
```
Dalej znajduje się słownik, który posłuży do zmieniania oznaczeń klas ze słów na liczby, wszystkie obiekty niebędące speedlimit zostają oznaczone jako other- 1.
```python
class_id_to_new_class_id = {'speedlimit': 0, 'stop': 1, 'crosswalk': 1, 'trafficlight': 1}
```
Deklaracja ścieżek do kolejno plików testowych i treningowych następuje w sposób następujący:
```python
images_path_train = Path('./train/images')
annotations_path_train = Path('./train/annotations')
images_path_test = Path('./test/images')
annotations_path_test = Path('./test/annotations')
```
Funkcja ```data_format(path,im_path)``` ma za zadanie odczyt danych z plików .xml, dotyczących zdjęć. Jej argument to ścieżka do katalogu z adnotacjami oraz ścieżka do katalogu ze zdjęciami. Funkcja ta sprawdza, czy na zdjęciu występuje znak speedlimit, czy nie. Jeżeli występuje przynajmniej jeden, to całe zdjęcie oznaczane jest jako speedlimit, jeżeli nie to oznaczane jest klasą pierwszego obiektu, co nie ma znaczenia, gdyż w dalszym etapie, wszystkie inne zostaną oznaczone jako klasa other. Funkcja sprawdza również, czy obszar zajmowany przez znak to 1/100 pola całego zdjęcia. Funkcja zwraca dane w formacie pandas.DataFrame.

Funkcja ```load_data(path,filename)``` odczytuje dane z pliku .csv i odpowiednio je modyfikuje. Argumentami funkcji są ścieżka do pliku oraz jego nazwa. Funkcja zwraca listę słowników zawierających przyporządkowany obraz, jego klasę oraz nazwę.

Funkcja ```learn(data)``` uczy słownik cv2.BOWKMeansTrainer. Wykorzystuje algorytm SIFT( skaloniezmiennicze przekształcenie cech). Argumentem funkcji są wcześniej przygotowane dane. Funkcja tworzy plik voc.npy ze słownikiem zawierającym dla każdej próbki obraz oraz etykietę klasy.

Funkcja ```extract_features(data)``` wyodrębnia cechy z przygotowanych wcześniej danych, podanych jako argument. Wykorzystuje ona FLANN( Fast Library for Approximate Nearest Neighbors) oraz obliczanie deskryptorów obraz z wykorzystaniem metod klasy BOWImgDescriptorExtractor pakietu csv. Funkcja zwraca dalej przetworzone dane, które będą wykorzystane do treningu.

Funkcja ```train(data)``` wykonuje trening i zwraca wytrenowany Random Forest classifier.

Funkcja ```predict(rf, data,n)``` dodaje do podanych danych hasło do słownika, którego etykietą jest otrzymana etykieta klasy, która zostaje przyporządkowana przez wytrenowany Random Forest classifier. Argumenty funkcji to Random Forest classifier, który został wcześniej poddany treningowi, dane testowe, wcześniej przygotowane, poprzez wyodrębnienie cech oraz liczbę, która określa dla ilu obiektów, ma być wykonana funkcja.

Funkcja ```evaluate(data,n)``` wykorzystywana jest w celach testowych. Wylicza ona wskaźniki jakości zadania klasyfikacji i zwraca ich wartości. Są to odpowiednio accuracy, precision oraz recall. Jej argumentami są dane, już po etapie predykcji oraz ilość danych, dla których ją wykonano.

Funkcje ```display_data(data,n)``` oraz ```print_evaluate_data(data)``` służa odpowiednio do wyświetlania wyników klasyfikacji oraz wskaźników jakości tego procesu.

W kodzie znaleźć można również ```test_main()```, który zawiera wywołania poszczególnych funkcji i cały proces klasyfikacji. Jest to funkcja testowa, która została w kodzie w celu testowania rozwiązań. Nie jest ona używana.

Funkcją główną, która wykonuje cały proces klasyfikacji, jest ```main()```. Pierwsze w kolejności jest wczytanie danych treningowych oraz zmiana ich formatu na plik .csv. Dalej w kolejności następuje nauka, wyodrębnianie cech oraz trening z wykorzystaniem odpowiednich funkcji. Następnie wyświetlany jest komunikat, który nakazuje wpisanie jednego z dwóch poleceń:

```detect```- klasyfikacja zostanie wykonana na wszystkich plikach ze zbioru treningowego.

```classify```- klasyfikacja zostanie wykonana na takiej ilości plikach, jaka zostanie podana przez użytkownika. Jeżeli liczba podana, byłaby większa niż liczba plików w katalogu trenigowym, program zakończy się komunikatem o za dużej liczbie podnej.

# Podsumowanie

W ramach projektu zrealizowano jedynie zadanie klasyfikacji zdjęć, nie wykonano detekcji znaków. Nie sprawdzano więc wycinków obrazów, podanych przez użytkownika, pod kątem występowania tam obrazów. Dane wyświetlane są użytkownikowi w takiej kolejności, w jakiej zostały zapisane, nie udało się ich posortować względem nazwy. Nie wykonywana jest również klasyfikacja kilku znaków z tego samego zdjęcia, gdyż przyjęto kryterium "ważności znaków speedlimit" opisywane w pierwszym akapicie. 




