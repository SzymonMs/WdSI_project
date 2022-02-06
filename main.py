import random
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas
import os
import xml.etree.ElementTree as ET
from pathlib import Path

images_path = Path('C:/Users/szymo/Desktop/Zadania/Semestr 5/Wprowadzenie do sztucznej inteligencji/Laby/Lab5/13_traffic_signs_cln/images')
annotations_path = Path('C:/Users/szymo/Desktop/Zadania/Semestr 5/Wprowadzenie do sztucznej inteligencji/Laby/Lab5/13_traffic_signs_cln/annotations')

def load_data(annotations_path):
    annotations=[os.path.join(directory_path, f) for directory_path, directory_name,
            files in os.walk(annotations_path) for f in files if f.endswith('.xml')]
    data_list = []
    for file in annotations:
        root = ET.parse(file).getroot()
        data = {}
        data['filename'] = Path(str(images_path) + '/'+ root.find("./filename").text)
        data['width'] = root.find("./size/width").text
        data['height'] = root.find("./size/height").text
        data['class'] = root.find("./object/name").text
        data['xmin'] = int(root.find("./object/bndbox/xmin").text)
        data['ymin'] = int(root.find("./object/bndbox/ymin").text)
        data['xmax'] = int(root.find("./object/bndbox/xmax").text)
        data['ymax'] = int(root.find("./object/bndbox/ymax").text)
        data_list.append(data)
    return data_list

def train_set(data):
    return pandas.DataFrame(data)

def main():
    train = train_set(load_data(annotations_path))
    class_id_to_new_class_id = {'speedlimit': 0, 'stop': 1, 'crosswalk': 1, 'trafficlight': 1}
    train['class'] = train['class'].apply(lambda x: class_id_to_new_class_id[x])
    print(train['class'].value_counts())

if __name__ == '__main__':
    main()