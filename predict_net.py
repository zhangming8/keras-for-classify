# -*- coding: utf-8 -*-

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import os
import imutils
import cv2
import time


label_dict = {'33020002':'0', '33020011':'1', '33020024':'2', '33030005':'3', '33050014':'4', '33050019':'5',
              '33050031':'6', '33050032':'7', '33050035':'8', '33050036':'9', '33060009':'10', '33070002':'11'}

norm_size = 32
save_model = './lenet.h5'  # save model name
image_name = '1-15-20.jpg'
show = False
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # use gpu number

    
def predict():
    # load the trained convolutional neural network
    print("loading network...")
    model = load_model(save_model)

    # start time
    start_time = time.time()
    #load the image
    image = cv2.imread(image_name)

    # pre-process the image for classification
    orig = image
    image = cv2.resize(image, (norm_size, norm_size))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
     
    # classify the input image
    result = model.predict(image)[0]
    cost_time = time.time() - start_time
    print("cost time", cost_time)

    # predict label
    proba = np.max(result)
    label_raw = str(np.where(result == proba)[0][0])
    label = list(label_dict.keys())[list(label_dict.values()).index(label_raw)]
    label = "{} is {} --prob: {:.2f}%".format(image_name, label, proba * 100)
    print(label)
    
    if show:   
        # draw the label on the image
        output = imutils.resize(orig, width=400)
        cv2.putText(output, label, (10, 25),cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)       
        # show the output image
        cv2.imshow("Output", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    predict()
