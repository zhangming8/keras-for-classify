import tensorflow as tf
import numpy as np
import cv2
# from keras.applications.resnet50 import preprocess_input
import time

# classes = ['33020002', '33050019', '33020011', '33070002', '33020024', '33050031', '33030005', '33050036', '33050032', '33050035', '33050014', '33060009']
label_dict = {'33020002':'0', '33020011':'1', '33020024':'2', '33030005':'3', '33050014':'4', '33050019':'5',
              '33050031':'6', '33050032':'7', '33050035':'8', '33050036':'9', '33060009':'10', '33070002':'11'}

def preprocess_input(img):
    # x = img[..., ::-1]
    # mean = [103.939, 116.779, 123.68]
    # std = None
    # x[..., 0] -= mean[0]
    # x[..., 1] -= mean[1]
    # x[..., 2] -= mean[2]
    # if std is not None:
    #     x[..., 0] /= std[0]
    #     x[..., 1] /= std[1]
    #     x[..., 2] /= std[2]
    x = img / 255.0
    return x


def recognize(jpg_path, pb_file_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        print('-----:' , pb_file_path)
        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")


        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            op = sess.graph.get_operations()

            for m in op:
                print(m.values())

            input_x = sess.graph.get_tensor_by_name("conv2d_1_input:0")
            # print input_x

            out_softmax = sess.graph.get_tensor_by_name("activation_4/Softmax:0")

            # print out_softmax

            for i in jpg_path:
                img = cv2.imread(i , 1)

                start_time = time.time()
                img = cv2.resize(img , (32 , 32))
                img = img.astype('float32')

                x = np.expand_dims(img, axis=0)
                img = preprocess_input(x)
                img_out_softmax = sess.run(out_softmax,
                                           feed_dict={input_x: img})

                end_time = time.time() - start_time
                print('cost time:' , end_time)

            # img = cv2.imread(jpg_path, 1)
            # img = cv2.resize(img, (32, 32))
            # img = img.astype('float32')
            # x = np.expand_dims(img, axis=0)
            #
            # img = preprocess_input(x)
            # img_out_softmax = sess.run(out_softmax,
            #                            feed_dict={input_x: img})
            #
            # print "img_out_softmax:", img_out_softmax
            # prediction_labels = np.argmax(img_out_softmax, axis=1)
            # print "label:", int(prediction_labels)
            # label = list(label_dict.keys())[list(label_dict.values()).index(str(int(prediction_labels)))]
            # label = "{} is {} --prob: {:.2f}%".format(jpg_path.split('/')[-1], label, np.max(img_out_softmax) * 100)
            # print(label)



pb_path = './tensorflow_model/tensor_model.pb'
img = './1-15-20.jpg'

imgs = [img] * 100

recognize(imgs, pb_path)