from sklearn.model_selection import train_test_split
import numpy as np

img_data_path = "npys/ForPretr_imgs.npy"
label_data_path= "npys/ForPretr_labels.npy"

img_npy = np.load(img_data_path)
label_npy = np.load(label_data_path)
print(len(label_npy))
img_rows = img_npy.shape[1]
img_cols = img_npy.shape[2]

train_img, val_img, train_label, val_label = train_test_split(img_npy, label_npy, test_size = 0.5, random_state=602625)
test_img, val_img, test_label, val_label = train_test_split(val_img, val_label, test_size = 0.4, random_state=602625)

np.save("npys/For_Pretraining/Train_imgs.npy", train_img)
np.save("npys/For_Pretraining/Train_labels.npy", train_label)

np.save("npys/For_Pretraining/Val_imgs.npy", val_img)
np.save("npys/For_Pretraining/Val_labels.npy", val_label)

np.save("npys/For_Pretraining/Test_imgs.npy", test_img)
np.save("npys/For_Pretraining/Test_labels.npy", test_label)

print("All Data is Ready to Train")
