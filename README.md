# WdSI_project

Projekt na zaliczenie przedmiotu Wprowadzenie do Sztucznej inteligencji. Zadaniem programu jest klasyfikacja znaków drogowych na dwie grupy: speedlimit oznaczone jako 0 oraz other oznaczone jako 1. W projekcie nie wykonano zadania detekcji znaku. 
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

# ZMIANY

Funkcja ```loadData(boolean)``` ma za zadanie wczytanie danych, w zależności od wartości zmiennej logicznej wczytuje on odpowiednie dane. Jeżeli jest ```True``` to wczytuje on dane treningowe, w przeciwnym wypadku testowe. Funkcja zwraca listę słowników z danymi.

Funkcja  ```Classify(data)``` służy do wyświetlenia w odpowiednim formacie wyników klasyfikacji.

Funkcja ```main()``` zaczyna się od przyjęcia polecenia sterującego. Następne jest przygotowanie danych. W dalszej kolejności następuje przyjęcie przez program liczby klasyfikowanych plików, ich nazwy, ilość wycinków z tych plików oraz ich współrzędne. Program dokonuje uczenia, a następnie klasyfikacji i testowania.

# Podsumowanie

W ramach projektu zrealizowano jedynie zadanie klasyfikacji zdjęć, nie wykonano detekcji znaków. Nie sprawdzano więc wycinków obrazów, podanych przez użytkownika, pod kątem występowania tam obrazów. Dane wyświetlane są użytkownikowi w takiej kolejności, w jakiej zostały zapisane, nie udało się ich posortować względem nazwy. Nie wykonywana jest również klasyfikacja kilku znaków z tego samego zdjęcia, gdyż przyjęto kryterium "ważności znaków speedlimit" opisywane w pierwszym akapicie. 




