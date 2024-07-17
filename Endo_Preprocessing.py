import cv2
import os

import matplotlib.pyplot as plt
import numpy as np

MT_0 = 0
MT_1 = 0


def Contour_process(IMAGE):
    imgcopy = IMAGE.copy()
    img_gray = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(img_gray, 35, 255, cv2.THRESH_BINARY)
    # thr = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    contours1,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_dis = 0
    y_dis = 0
    Cnum = 0
    for k in range(len(contours1)):
        x, y, w, h = cv2.boundingRect(contours1[k])
        print("Contour_num= ", k, "x_dis= ", w, "y_dis= ", h)
        if x_dis < w:
            if y_dis < h:
                x_dis, y_dis = w - x, h - y
                x_1, y_1, w_1, h_1 = x, y, w, h
                Cnum = k

    stencil_r = np.zeros(IMAGE.shape[:-1]).astype(np.uint8)
    stencil_g = np.zeros(IMAGE.shape[:-1]).astype(np.uint8)
    stencil_b = np.zeros(IMAGE.shape[:-1]).astype(np.uint8)
    stencil = cv2.merge([stencil_r, stencil_g, stencil_b])

    cv2.drawContours(IMAGE, contours1[Cnum], -1, (0, 0, 255), 3)
    (i, j), r = cv2.minEnclosingCircle(contours1[Cnum])
    stencil = cv2.circle(stencil, (int(i), int(j)), int(r) - 21, (255, 255, 255), -1)
    draw_ellipse = stencil.copy()
    ellipse_gray = cv2.cvtColor(draw_ellipse, cv2.COLOR_BGR2GRAY)
    contours2, _ = cv2.findContours(ellipse_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    x_2, y_2, w_2, h_2 = cv2.boundingRect(contours2[0])
    sel = stencil != 0  # select everything that is not mask_value
    stencil[sel] = imgcopy[sel]  # and fill it with fill_color
    crop = stencil[y_2:y_2 + h_2, x_2:x_2 + w_2]
    return crop


def processing(Hospital_name, CLASS, DETAILS, data_path, LIST):
    global Be
    global Ma
    global No
    global nP
    processed_list = os.listdir("data/by hospital/Processed_img")
    LIST = list(set(LIST).intersection(processed_list))
    if CLASS == "Benign":
        Be = Be + len(LIST)
    elif CLASS == "Malignant":
        Ma = Ma + len(LIST)
    elif CLASS == "Normal":
        No = No + len(LIST)
    elif CLASS == "NP":
        nP = nP + len(LIST)

    image_shape = []
    N = 0
    for image_name in LIST:
        image = cv2.imread(data_path + image_name, cv2.IMREAD_COLOR)
        print("*" * 30)
        print(data_path + image_name)
        print("*" * 30)
        pred_img = Contour_process(image)
        image_name = image_name.split(".")[0]+".png"
        if DETAILS == "None":
            cv2.imwrite("data/processed_data/" + CLASS + "/" + image_name, pred_img)
        else:
            cv2.imwrite("data/processed_data/" + CLASS + "/" + DETAILS + "/" + image_name, pred_img)
        image_shape.append(image.shape)
        N = N + 1
    a = set(image_shape)
    # cv2.imshow("a",image)
    # cv2.waitKey(10)
    print('*' * 30)
    print(Hospital_name + CLASS + ' Done')
    print(a)
    print('*' * 30)
    print(Be, Ma, No, nP)
    print('*' * 30)
    return Be, Ma, No, nP


def read_data_path(PATH):
    Class_list = ['Benign', 'Malignant', 'NP', 'Normal']
    folder_list = os.listdir(PATH)
    folder_list = sorted(folder_list)
    print(folder_list)
    for folder_name in folder_list:
        Classes_folder = os.listdir(PATH + '/' + folder_name + '/')
        print(folder_name)
        print(Classes_folder)
        for Class in Class_list:
            if Class in Classes_folder:
                if Class == 'Benign':
                    Benign_list = ['IP', 'Non-vascular, etc', 'Vascular']
                    for Benign_Class in Benign_list:
                        Benign_path = PATH + '/' + folder_name + '/' + Class + '/' + Benign_Class + '/'
                        file_list = os.listdir(Benign_path)
                        processing(folder_name, "Benign", Benign_Class, Benign_path, file_list)
                elif Class == 'Malignant':
                    Malignant_list = ['Epithelial', 'Non-Epithelial']
                    for Malignant_Class in Malignant_list:
                        Malignant_path = PATH + '/' + folder_name + '/' + Class + '/' + Malignant_Class + '/'
                        file_list = os.listdir(Malignant_path)
                        processing(folder_name, "Malignant", Malignant_Class, Malignant_path, file_list)
                elif Class == 'NP':
                    NP_path = PATH + '/' + folder_name + '/' + Class + '/'
                    file_list = os.listdir(NP_path)
                    processing(folder_name, "NP", "None", NP_path, file_list)
                elif Class == 'Normal':
                    NM_path = PATH + '/' + folder_name + '/' + Class + '/'
                    file_list = os.listdir(NM_path)
                    processing(folder_name, "Normal", "None", NM_path, file_list)
    Before_val = [759, 430, 1974, 1433]
    After_val = [Be, Ma, No, nP]
    data = {'Benign': Be, 'Malignant': Ma, 'Normal': No, 'NP': nP}
    categories = list(data.keys())
    val = list(data.values())
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.bar(categories, Before_val, label="Before")
    ax.bar(categories, After_val, label="After")
    ax.set_title("Data size after preprocessing")
    ax.legend()
    plt.show()
    return folder_list

def Show_data_volume():
    Before_val = [759, 430, 1974, 1433]
    After_val = [Be, Ma, No, nP]
    benign = len(os.listdir("Benign/"))
    malignant = len(os.listdir("Malignant/"))
    normal = len(os.listdir("Normal/"))
    NP = len(os.listdir("NP/"))
    Augmentation = [benign, malignant, normal, NP]
    data = {'Benign': benign, 'Malignant': malignant, 'Normal': normal, 'NP': NP}
    categories = list(data.keys())
    val = list(data.values())
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.bar(categories, Augmentation, label="Aug")
    ax.bar(categories, Before_val, label="Before")
    ax.bar(categories, After_val, label="After")
    ax.set_title("Data size after preprocessing")
    ax.legend()
    plt.show()
    print("Benign = ", benign, "Malignant = ", malignant, "Normal = ", normal, "NP = ", NP)


if __name__ == '__main__':
    folder_dir = "data/by hospital/Original"
    read_data_path(folder_dir)
    Folder_list = os.listdir("data/processed_data/")
    for folder in Folder_list:
        Data_Augmentation(folder)
    Show_data_volume()
