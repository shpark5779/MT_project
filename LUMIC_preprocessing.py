import csv
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# print(os.listdir("labeled-images/lower-gi-tract/pathological-findings"))
tv_class_list = os.listdir("LIMUC/train_and_validation_sets/")
tst_class_list = os.listdir("LIMUC/test_set/")
tv_path = "LIMUC/train_and_validation_sets/"
tst_path = "LIMUC/test_set/"
print(tv_class_list)
NOI = 0
for label in tv_class_list:
    print(label)
    LN = len(os.listdir("LIMUC/train_and_validation_sets/"+label))
    if label == 'Mayo 3':
        NOI += (LN*4)
    elif label == 'Mayo 2':
        NOI += (LN*2)
    else:
        NOI += LN
    print(LN)
    print(NOI)

patho_img_list = []
name_label = []

sharpen_f = np.array([[-1, -1, -1, -1, -1],
                      [-1, 2, 2, 2, -1],
                      [-1, 2, 9, 2, -1],
                      [-1, 2, 2, 2, -1],
                      [-1, -1, -1, -1, -1]])/9.0

imgs = np.ndarray(shape=(NOI,288,352,3))
labels = np.ndarray(shape=(NOI))

i = 0
L = 3
for label in tv_class_list:
    file_name = os.listdir(tv_path+label)
    for file in file_name:
        if label == 'Mayo 3':
            img = cv2.imread(tv_path+label+'/'+file)
            f = cv2.filter2D(img, -1, sharpen_f)
            img_rr = cv2.rotate(img, cv2.ROTATE_180)
            ff = cv2.filter2D(img_rr, -1, sharpen_f)
            imgs[i] = img
            imgs[i+1] = f
            imgs[i+2] = img_rr
            imgs[i+3] = ff
            labels[i:i+4] = L
            i+=4
            if i % 100 == 0:
                print("{0}/{1} images Done.".format(i,len(labels)))
        elif label == 'Mayo 2':
            img = cv2.imread(tv_path+label+'/'+file)
            f = cv2.filter2D(img, -1, sharpen_f)
            imgs[i] = img
            imgs[i+1] = f
            labels[i:i+2] = L
            i+=2
            if i % 100 == 0:
                print("{0}/{1} images Done.".format(i,len(labels)))
        else:
            img = cv2.imread(tv_path+label+'/'+file)
            labels[i] = L
            i+=1
            if i % 100 == 0:
                print("{0}/{1} images Done.".format(i,len(labels)))
    L-=1
    

print(i)
print(len(labels), np.unique(labels))
print(labels)
print("ALL images Done.")
np.save("npys/LUMIC_Train_imgs.npy", imgs)
np.save("npys/LUMIC_Train_labels.npy", labels)
