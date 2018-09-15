from __future__ import print_function
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import random
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical, plot_model
from keras.callbacks import LearningRateScheduler
import keras.backend as K
from imutils import paths
from mynet import AlexNet, LeNet, LeNet2

label_dict = {'33020002': '0', '33020011': '1', '33020024': '2', '33030005': '3', '33050014': '4', '33050019': '5',
              '33050031': '6', '33050032': '7', '33050035': '8', '33050036': '9', '33060009': '10', '33070002': '11'}

# parameters setting
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # use gpu number
file_path = '/Users/ming/Downloads/dataset/use_sku'  # data file path
train_ratio = 0.7  # train ratio
EPOCHS = 10  # epoch
INIT_LR = 1e-3  # initial learning rate
lr_decay_interval = 30
lr_decay_ratio = 0.1  # lr * lr_decay_ratio every lr_decay_interval epoch
Batch_Size = 16  # batch size
CLASS_NUM = 12
norm_size = 32  # resize images to norm_size * norm_size
save_model = './lenet.h5'  # save model name


def load_data(Paths):
    data = []
    labels = []
    for imagePath in Paths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (norm_size, norm_size))
        image = img_to_array(image)
        # data
        data.append(image)
        # label
        label = int(label_dict[imagePath.split(os.path.sep)[-2]])
        labels.append(label)

    # scale the pixel value [0,255] to [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # convert the labels from integers to vectors
    labels = to_categorical(labels, num_classes=CLASS_NUM)
    print(labels)
    return data, labels


def scheduler(epoch):
    if epoch % lr_decay_interval == 0:
        lr = K.get_value(model.optimizer.lr)
        if epoch == 0:
            print('lr: ', lr)
        else:
            K.set_value(model.optimizer.lr, lr * lr_decay_ratio)
            print('lr: ', lr * lr_decay_ratio)
    return K.get_value(model.optimizer.lr)


if __name__ == '__main__':
    # get all the images paths
    imagePaths = sorted(list(paths.list_images(file_path)))
    print('all image number is', len(imagePaths))
    # shuffle the order of images
    random.seed(20)
    random.shuffle(imagePaths)
    train_num = int(train_ratio * len(imagePaths))
    # split the train and test images
    train_file_path = imagePaths[:train_num]
    print('train num is', len(train_file_path))
    test_file_path = imagePaths[train_num:]
    print('test num is', len(test_file_path))
    # data ,label of train ,test
    trainX, trainY = load_data(train_file_path)
    testX, testY = load_data(test_file_path)
    # data augmentation
    aug = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")

    # initialize the model
    print("compiling model...")
    model = LeNet(width=norm_size, height=norm_size, depth=3, classes=CLASS_NUM)
    # model = AlexNet(width=norm_size, height=norm_size, depth=3, classes=CLASS_NUM)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    # model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file='model.png')
    # train the network
    print("training network...")
    lr_decay = LearningRateScheduler(scheduler)
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=Batch_Size),
                            validation_data=(testX, testY), steps_per_epoch=len(trainX) // Batch_Size,
                            epochs=EPOCHS, verbose=1, callbacks=[lr_decay])

    # save the model
    print("save model...")
    model.save(save_model)

    # plot the training loss and accuracy
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="center right")
    plt.savefig('acc_loss.png')

