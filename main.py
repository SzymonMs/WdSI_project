import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import csv

# TODO Jakość kodu i raport (3/4)


# TODO Skuteczność klasyfikacji 0.0 (0/4)
# TODO [0.00, 0.50) - 0.0
# TODO [0.50, 0.55) - 0.5
# TODO [0.55, 0.60) - 1.0
# TODO [0.60, 0.65) - 1.5
# TODO [0.65, 0.70) - 2.0
# TODO [0.70, 0.75) - 2.5
# TODO [0.75, 0.80) - 3.0
# TODO [0.80, 0.85) - 3.5
# TODO [0.85, 1.00) - 4.0

# stderr:
# Traceback (most recent call last):
#   File "main.py", line 234, in <module>
#     main()
#   File "main.py", line 215, in main
#     learn(data_train)
#   File "main.py", line 75, in learn
#     voc=bow.cluster()
# cv2.error: OpenCV(4.5.4) /tmp/pip-req-build-th1mncc2/opencv/modules/features2d/src/bagofwords.cpp:94: error: (-215:Assertion failed) !descriptors.empty() in function 'cluster'

# TODO Skuteczność detekcji (0/2)

class_id_to_new_class_id = {'speedlimit': 0, 'stop': 1, 'crosswalk': 1, 'trafficlight': 1}

images_path_train = Path('./train/images')
annotations_path_train = Path('./train/annotations')
images_path_test = Path('./test/images')
annotations_path_test = Path('./test/annotations')

def data_format(path,im_path):
    data_list=[]
    annotations=[os.path.join(directory_path, f) for directory_path, directory_name,
            files in os.walk(path) for f in files if f.endswith('.xml')]
    for file in annotations:
        root = ET.parse(file).getroot()
        class_list=[]
        for obj in root.findall('object'):
            data = {}
            data['filename'] = Path(str(im_path) + '/'+ root.find("./filename").text)
            #data['filename'] = root.find("./filename").text
            data['width'] = int(root.find("./size/width").text)
            data['height'] = int(root.find("./size/height").text)
            data['class'] = obj.find("./name").text
            data['xmin'] = int(obj.find("./bndbox/xmin").text)
            data['xmax'] = int(obj.find("./bndbox/xmax").text)
            data['ymin'] = int(obj.find("./bndbox/ymin").text)
            data['ymax'] = int(obj.find("./bndbox/ymax").text)
            class_list.append(data['class'])
        # TODO Zadanie klasyfikacji polegalo na klasyfikacji fragmentow zdjec, a nie calych zdjec.
        if class_list.count("speedlimit")>0:
            data['class']="speedlimit"
        else:
            data['class']=class_list[0]
        class_list.clear()
        if ((data['xmax']-data['xmin'])*(data['ymax']-data['ymin']))>=0.01*data['width']*data['height']:
            data_list.append(data)
        else:
            continue
    #data_list.sort()
    DataFrame=pandas.DataFrame(data_list)
    return DataFrame

def load_data(path,filename):
    data_list=pandas.read_csv(os.path.join(path,filename))
    data=[]
    class_to_num={}
    for index,files in data_list.iterrows():
        id=class_id_to_new_class_id[files['class']]
        img_name=files['filename']
        image=cv2.imread(os.path.join(path,img_name))
        data.append({'image':image,'label':id,'filename':img_name})
        if id not in class_to_num:
            class_to_num[id]=0
        class_to_num[id]+=1
    class_to_num=dict(sorted(class_to_num.items(),key=lambda item: item[0]))
    #print(class_to_num)
    return data

def learn(data):
    size=128
    bow=cv2.BOWKMeansTrainer(size)
    sift=cv2.SIFT_create()
    for files in data:
        k=sift.detect(files['image'],None)
        k,desc=sift.compute(files['image'],k)
        if desc is not None:
            bow.add(desc)
    voc=bow.cluster()
    np.save('voc.npy',voc)

def extract_features(data):
    size=128
    sift = cv2.SIFT_create()
    flann_base_matcher = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann_base_matcher)
    vocabulary = np.load('voc.npy')
    bow.setVocabulary(vocabulary)

    for files in data:
        k = sift.detect(files['image'], None)
        imgDes = bow.compute(files['image'], k)
        if imgDes is None:
            files['desc'] = np.zeros((1, size))
        else:
            files['desc'] = imgDes
    return data

def train(data):
    size=128
    desc_matrix = np.empty((1, size))
    label_vector = []
    for files in data:
        if files['desc'] is None:
            continue
        else:
            label_vector.append(files['label'])
            desc_matrix = np.vstack((desc_matrix, files['desc']))
    clf = RandomForestClassifier(size)
    clf.fit(desc_matrix[1:], label_vector)
    return clf

def predict(rf, data,n):
    i=0
    for files in data:
        if i<n:
            if files['desc'] is None:
                continue
            else:
                files.update({'label_pred': rf.predict(files['desc'])[0]})
        i=i+1
    return data

def evaluate(data,n):
    i=0
    y_pred = [0,0,0,0]
    y_real = [0,0,0,0]
    for files in data:
        if i<n:
            y_pred.append(files['label_pred'])
            y_real.append(files['label'])
        i=i+1
    tn, fp, fn, tp = confusion_matrix(y_pred, y_real, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)*100
    precision=tp/(tp+fp)*100
    recall=tp/(tp+fn)*100
    return accuracy,precision,recall

def display_data(data,n):
    i=0
    for files in data:
        if i<n:
            #print(files['filename'],"wykryta klasa: ", files['label_pred'], "prawdziwa klasa: ", files['label'])
            print(files['label_pred'])
        i=i+1

def print_evaluate_data(data):
    print("accuracy = ", format(data[0], '.4g'), "%")
    print("precision = ", format(data[1], '.4g'), "%")
    print("recall = ", format(data[2], '.4g'), "%")
    return

def test_main():
    DataFrame_train=data_format(annotations_path_train,images_path_train)
    DataFrame_train.to_csv("Train.csv")
    DataFrame_test = data_format(annotations_path_test, images_path_test)
    DataFrame_test.to_csv("Test.csv")
    data_train = load_data('./', 'Train.csv')
    train_count = len(data_train)
    print(train_count)
    data_test = load_data('./', 'Test.csv')
    test_count = len(data_test)
    print(test_count)
    print('learning BoVW')
    if os.path.isfile('voc.npy'):
        print('BoVW is already learned')
    else:
        learn(data_train)
    print('extracting train features')
    data_train = extract_features(data_train)
    print('training')
    rf = train(data_train)
    print('extracting test features')
    data_test = extract_features(data_test)
    print('testing')
    data_test = predict(rf, data_test,10)
    evaluate_data=evaluate(data_test,10)
    print_evaluate_data(evaluate_data)
    display_data(data_test,10)

def main():
    #print("Podaj polecenie 'detect' albo 'classify': ")
    command=input()
    if command=="detect":
        # print("Loading data...")
        DataFrame_train = data_format(annotations_path_train, images_path_train)
        DataFrame_train.to_csv("Train.csv")
        data_train = load_data('./', 'Train.csv')
        # print("Learning...")
        # if os.path.isfile('voc.npy'):
        #    #print('BoVW is already learned')
        # else:
        #    learn(data_train)
        learn(data_train)
        data_train = extract_features(data_train)
        rf = train(data_train)
        #print("Loading data...")
        DataFrame_test = data_format(annotations_path_test, images_path_test)
        DataFrame_test.to_csv("Test.csv")
        data_test = load_data('./', 'Test.csv')
        test_count = len(data_test)
        #print("Work in progress... ")
        # TODO To nie jest detekcja.
        data_test = extract_features(data_test)
        data_test = predict(rf, data_test,test_count)
        #print("List of files: ")
        display_data(data_test, test_count)
    elif command=="classify":
        n = input()
        n = int(n)
        # print("Loading data...")
        # TODO Niepotrzebne przepisywanie danych do nowego pliku.
        DataFrame_train = data_format(annotations_path_train, images_path_train)
        DataFrame_train.to_csv("Train.csv")
        data_train = load_data('./', 'Train.csv')
        # print("Learning...")
        # if os.path.isfile('voc.npy'):
        #    #print('BoVW is already learned')
        # else:
        #    learn(data_train)
        learn(data_train)
        data_train = extract_features(data_train)
        rf = train(data_train)
        #print("Loading data...")
        # TODO Informacje o zdjeciach do klasyfikacji sa podawane na standardowe wejscie.
        DataFrame_test = data_format(annotations_path_test, images_path_test)
        DataFrame_test.to_csv("Test.csv")
        data_test = load_data('./', 'Test.csv')
        test_count = int(len(data_test))
        #print("Podaj ile zdjec zaklasyfikowac wiedzac ze jest ich w sumie ",test_count,": ")
        if n<=test_count:
            #print("Work in progress... ")
            data_test = extract_features(data_test)
            data_test = predict(rf, data_test,n)
            #print("List of files: ")
            display_data(data_test,n)
        else:
            print("Podales za duza liczbe")

if __name__ == '__main__':
    main()