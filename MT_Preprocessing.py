import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def contour_process(Input):
    copy = Input.copy()
    rgb_img = cv2.cvtColor(Input, cv2.COLOR_BGR2RGB)
    gr_img = cv2.cvtColor(Input, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2. threshold(gr_img, 25, 255, cv2.THRESH_BINARY)
    
    contours1,_ =  cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_dis, y_dis, C_num = 0, 0, 0
    for k in range(len(contours1)):
        x,y,w,h = cv2.boundingRect(contours1[k])
        # print("Number:", k, "x_dis: ", w-x, "y_dis: ", h-y)
        if x_dis<w:
            if y_dis<h:
                x_dis, y_dis = w-x, h-y
                x_1, y_1, w_1, h_1 = x,y,w,h
                C_num = k
    stencil_r = np.zeros(Input.shape[:-1]).astype(np.uint8)
    stencil_g = np.zeros(Input.shape[:-1]).astype(np.uint8)
    stencil_b = np.zeros(Input.shape[:-1]).astype(np.uint8)
    rect = cv2.minAreaRect(contours1[C_num])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    a = cv2.drawContours(rgb_img, [box], -1, (0, 0, 255),10)

    croped_img = copy[np.min(box[:][1]):np.max(box[:][0]),np.min(box[:][0]):np.max(box)]
    rs_img = cv2.resize(croped_img,(256,256), cv2.INTER_CUBIC)

    return rs_img

if __name__ == '__main__':

  D_path1, D_path2  = "MT_0/", "MT_1/"
  
  mt0_list = os.listdir(D_path1)
  mt1_list = os.listdir(D_path2)
  j = 0
  for file_name in mt0_list:
      mt0_list[j]="MT_0/"+file_name
      j+=1
  j = 0
  for file_name in mt1_list:
      mt1_list[j]="MT_1/"+file_name
      j+=1
  print(mt0_list[0])
  num_mt0 = len(mt0_list)
  num_mt1 = len(mt1_list)
  
  print(num_mt0, num_mt1)
  imgs = np.ndarray(shape=(num_mt0+num_mt1,256,256,3))
  labels = np.ndarray(shape=(num_mt0+num_mt1))
  print(imgs.shape)
  img_list = mt0_list+mt1_list
  print(len(img_list))
  print("_"*30)
  print("Creat Img Dataset")
  print("_"*30)
  i = 0
  a = list
  for img_name in img_list:
      # print(img_name)
      img = cv2.imread(img_name)
      pred_img = contour_process(img)
      imgs[i] = pred_img
      if i <= len(mt0_list):
          labels[i] = 0
          if i == len(mt0_list):
              print(i)
              print("*"*20, "Convert", "*"*20)
      elif i > len(mt0_list):
          labels[i] = 1
  
      if i % 100 == 0:
          print("{0}/{1} images Done.".format(i,len(img_list)))
      i+=1
  print(np.unique(labels))
  print("ALL images Done.".format(i,len(img_list)))
  np.save("Total_imgs.npy", imgs)
  np.save("Total_imgs_labels.npy", labels)
