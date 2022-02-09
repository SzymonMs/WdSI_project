import random
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas
import os
import xml.etree.ElementTree as ET
from pathlib import Path

class_id_to_new_class_id = {'speedlimit': 0, 'stop': 1, 'crosswalk': 1, 'trafficlight': 1}

images_path = Path('./images')
annotations_path = Path('./annotations')

def read_xml(path):
    annotations=[os.path.join(directory_path, f) for directory_path, directory_name,
            files in os.walk(path) for f in files if f.endswith('.xml')]
    data_list = []
    for file in annotations:
        root = ET.parse(file).getroot()
        data = {}
        data['filename'] = Path(str(images_path) + '/'+ root.find("./filename").text)
        data['width'] = int(root.find("./size/width").text)
        data['height'] = int(root.find("./size/height").text)
        data['class'] = root.find("./object/name").text
        data['xmin'] = int(root.find("./object/bndbox/xmin").text)
        data['ymin'] = int(root.find("./object/bndbox/ymin").text)
        data['xmax'] = int(root.find("./object/bndbox/xmax").text)
        data['ymax'] = int(root.find("./object/bndbox/ymax").text)
        data_list.append(data)
    return data_list

def train_set(data):
    return pandas.DataFrame(data)

def xml_to_csv(path):
    data_list=[]
    annotations=[os.path.join(directory_path, f) for directory_path, directory_name,
            files in os.walk(path) for f in files if f.endswith('.xml')]
    for file in annotations:
        root = ET.parse(file).getroot()
        data = {}
        data['filename'] = Path(str(images_path) + '/'+ root.find("./filename").text)
        #data['filename'] = root.find("./filename").text
        data['width'] = int(root.find("./size/width").text)
        data['height'] = int(root.find("./size/height").text)
        data['class'] = root.find("./object/name").text
        data['xmin'] = int(root.find("./object/bndbox/xmin").text)
        data['ymin'] = int(root.find("./object/bndbox/ymin").text)
        data['xmax'] = int(root.find("./object/bndbox/xmax").text)
        data['ymax'] = int(root.find("./object/bndbox/ymax").text)
        data_list.append(data)
    DataFrame=pandas.DataFrame(data_list)
    DataFrame.to_csv("Train.csv")

def load_data(path,filename):
    data_list=pandas.read_csv(os.path.join(path,filename))
    data=[]
    class_to_num={}
    for index,files in data_list.iterrows():
        id=class_id_to_new_class_id[files['class']]
        img_path=files['filename']
        image=cv2.imread(os.path.join(path,img_path))
        data.append({'image':image,'label':id})
        if id not in class_to_num:
            class_to_num[id]=0
        class_to_num[id]+=1
    class_to_num=dict(sorted(class_to_num.items(),key=lambda item: item[0]))
    print(class_to_num)

    return data

def main():
    #train = train_set(read_xml(annotations_path))
    #train['class'] = train['class'].apply(lambda x: class_id_to_new_class_id[x])
    #print(train['class'].value_counts())

    xml_to_csv(annotations_path)
    train=load_data('./','Train.csv')

if __name__ == '__main__':
    main()