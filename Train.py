import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras import optimizers,metrics
from tensorflow.keras.callbacks import ModelCheckpoint, History, EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report, ConfusionMatrixDisplay



def xcept():
    API_model = Xception(include_top=False, weights=None, input_tensor=None,
                         input_shape=(288,352,3), pooling=None, classes=4)
    x = API_model.output
    x = GlobalAveragePooling2D()(x)
    output = Dense(4, activation='softmax')(x)
    model = Model(inputs=API_model.inputs, outputs=output)
    return model

def learning(train_i, train_l,val_i, val_l, C_weight):
    API_model = xcept()
    adam_w = optimizers.Adam()
    API_model.summary()
    API_model.compile(loss="categorical_crossentropy", optimizer=adam_w,
    metrics = [metrics.mae, metrics.categorical_accuracy])
    # tensorboard = TensorBoard(log_dir = "PreTrain_log/240719_16_35/", histogram_freq = 1, write_graph=True)
    CP = ModelCheckpoint("HDF5/LUMIC_240723.hdf5", monitor='loss', save_best_only=True)
    # ES = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1, mode="min",
    #                    restore_best_weights=True)
    API_model.fit(train_i, train_l,validation_data=(val_i, val_l), batch_size=256, epochs=100, verbose=1,
                  shuffle=True, callbacks=[CP], class_weight=C_weight)
    return API_model

def classes_weight(label):
    a=list(tr_label_data).count(0)
    b=list(tr_label_data).count(1)
    c=list(tr_label_data).count(2)
    d=list(tr_label_data).count(3)
    print('0= ', a, '1= ',b, '2= ',c, '3= ',d)
    weight_0 = (1/a)*tr_label_data.shape[0]/4
    weight_1 = (1/b)*tr_label_data.shape[0]/4
    weight_2 = (1/c)*tr_label_data.shape[0]/4
    weight_3 = (1/d)*tr_label_data.shape[0]/4

    CW = {0:weight_0,
          1:weight_1,
          2:weight_2,
          3:weight_3,}

    print('Weight for Classes 0: {:.2f}'.format(weight_0),
          'Weight for Classes 1: {:.2f}'.format(weight_1),
          'Weight for Classes 2: {:.2f}'.format(weight_2),
          'Weight for Classes 3: {:.2f}'.format(weight_3),)
    return CW

if __name__ == '__main__':
    # tr_img_data = np.load("npys/For_Pretraining/Train_imgs.npy")
    # tr_label_data = np.load("npys/For_Pretraining/Train_labels.npy")
    # v_img_data = np.load("npys/For_Pretraining/Val_imgs.npy")
    # v_label_data = np.load("npys/For_Pretraining/Val_labels.npy")
#     np.save("npys/LUMIC_Train_imgs.npy", imgs)
#     np.save("npys/LUMIC_Train_labels.npy", labels)

    img_data = np.load("npys/LUMIC_Train_imgs.npy")
    label_data = np.load("npys/LUMIC_Train_labels.npy")

    train_img, val_img, train_label, val_label = train_test_split(img_data, label_data, test_size = 0.3, random_state=602625)

    tr_c = tf.keras.utils.to_categorical(train_label)
    v_c = tf.keras.utils.to_categorical(val_label)

    CW = classes_weight(label_data)
    history = learning(train_img, tr_c, val_img, v_c, CW)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss graph')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train','val'], loc='upper left')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy graph')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train','val'], loc='upper left')
    plt.show()
