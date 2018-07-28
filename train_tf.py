# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'

resize = 32
batch_size = 16
num_class = 12
steps = 1000
lr = 0.001
keep_prob = 0.5

label_dict = {'33020002':'0', '33020011':'1', '33020024':'2', '33030005':'3', '33050014':'4', '33050019':'5',
              '33050031':'6', '33050032':'7', '33050035':'8', '33050036':'9', '33060009':'10', '33070002':'11'}


def get_files(data_dir):
    class_train = []
    label_train = []
    for folder in os.listdir(data_dir):
        for pic in os.listdir(data_dir+folder):
            class_train.append(data_dir+folder+'/'+pic)
            label_train.append(label_dict[folder])
    temp = np.array([class_train,label_train])
    temp = temp.transpose()
    #shuffle the samples
    np.random.shuffle(temp)
    #after transpose, images is in dimension 0 and label in dimension 1
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in label_list]
    #print(label_list)
    return image_list,label_list


def get_batches(image,label,resize_w,resize_h,batch_size,capacity):
    #convert the list of images and labels to tensor
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int64)
    queue = tf.train.slice_input_producer([image,label])
    label = queue[1]
    image_c = tf.read_file(queue[0])
    image = tf.image.decode_jpeg(image_c,channels = 3)
    #resize
    image = tf.image.resize_image_with_crop_or_pad(image,resize_w,resize_h)
    #(x - mean) / adjusted_stddev
    image = tf.image.per_image_standardization(image)
    
    image_batch,label_batch = tf.train.batch([image,label],
                                             batch_size = batch_size,
                                             num_threads = 64,
                                             capacity = capacity)
    images_batch = tf.cast(image_batch,tf.float32)
    labels_batch = tf.reshape(label_batch,[batch_size])
    return images_batch,labels_batch


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev = 0.01))


# init weights
weights = {
    "w1":init_weights([3,3,3,16]),
    "w2":init_weights([3,3,16,128]),
    "w3":init_weights([3,3,128,256]),
    "w4":init_weights([4096,4096]),
    "w_out":init_weights([4096,num_class])
    }

# init biases
biases = {
    "b1":init_weights([16]),
    "b2":init_weights([128]),
    "b3":init_weights([256]),
    "b4":init_weights([4096]),
    "b_out":init_weights([num_class])
    }


def conv2d(x,w,b):
    x = tf.nn.conv2d(x,w,strides = [1,1,1,1],padding = "SAME")
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)


def pooling(x):
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = "SAME")


def norm(x,lsize = 4):
    return tf.nn.lrn(x,depth_radius = lsize,bias = 1,alpha = 0.001/9.0,beta = 0.75)


def model(image_batches):
    l1 = conv2d(image_batches,weights["w1"],biases["b1"])
    l2 = pooling(l1)
    l2 = norm(l2)
    l3 = conv2d(l2,weights["w2"],biases["b2"])
    l4 = pooling(l3)
    l4 = norm(l4)
    l5 = conv2d(l4,weights["w3"],biases["b3"])
    #same as the batch size
    l6 = pooling(l5)
    l6 = tf.reshape(l6,[-1,weights["w4"].get_shape().as_list()[0]])
    l7 = tf.nn.relu(tf.matmul(l6,weights["w4"])+biases["b4"])
    soft_max = tf.add(tf.matmul(l7,weights["w_out"]),biases["b_out"])
    return soft_max


def model2(image_batches):
    conv1_weights = init_weights([5, 5, 3, 32])
    conv1_biases = init_weights([32])
    conv1 = tf.nn.conv2d(image_batches, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
    conv2_weights = init_weights([5, 5, 32, 64])
    conv2_biases = init_weights([64])
    conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu2 = tf.nn.relu( tf.nn.bias_add(conv2, conv2_biases) )  
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
    fc1_weights = init_weights([7 * 7 * 64, 1024])
    fc1_baises = init_weights([1024])
    pool2_vector = tf.reshape(pool2, [-1, 7 * 7 * 64])  
    fc1 = tf.nn.relu(tf.matmul(pool2_vector, fc1_weights) + fc1_baises)  
    fc1_dropout = tf.nn.dropout(fc1, keep_prob)  
    fc2_weights = init_weights([1024, num_class])
    fc2_biases = init_weights([num_class])
    fc2 = tf.matmul(fc1_dropout, fc2_weights) + fc2_biases  
    y_conv = tf.nn.softmax(fc2) 
    return y_conv

def model3(image_batches):
    # 第一层：卷积层
    # 过滤器大小为5*5, 当前层深度为1， 过滤器的深度为32
    conv1_weights = tf.get_variable("conv1_weights", [5, 5, 3, 32], initializer=tf.truncated_normal_initializer(stddev=0.01))
    conv1_biases = tf.get_variable("conv1_biases", [32], initializer=tf.constant_initializer(0.01))
    # 移动步长为1, 使用全0填充
    conv1 = tf.nn.conv2d(image_batches, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    # 激活函数Relu去线性化
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
      
    #第二层：最大池化层  
    #池化层过滤器的大小为2*2, 移动步长为2，使用全0填充  
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
      
    #第三层：卷积层  
    conv2_weights = tf.get_variable("conv2_weights", [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.01)) #过滤器大小为5*5, 当前层深度为32， 过滤器的深度为64  
    conv2_biases = tf.get_variable("conv2_biases", [64], initializer=tf.constant_initializer(0.01))  
    conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME') #移动步长为1, 使用全0填充  
    relu2 = tf.nn.relu( tf.nn.bias_add(conv2, conv2_biases) )  
      
    #第四层：最大池化层  
    #池化层过滤器的大小为2*2, 移动步长为2，使用全0填充  
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
      
    #第五层：全连接层  
    fc1_weights = tf.get_variable("fc1_weights", [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.01)) #7*7*64=3136把前一层的输出变成特征向量  
    fc1_baises = tf.get_variable("fc1_baises", [1024], initializer=tf.constant_initializer(0.01))  
    pool2_vector = tf.reshape(pool2, [-1, 7 * 7 * 64])  
    fc1 = tf.nn.relu(tf.matmul(pool2_vector, fc1_weights) + fc1_baises)  
      
    #为了减少过拟合，加入Dropout层  
    fc1_dropout = tf.nn.dropout(fc1, keep_prob)  
      
    #第六层：全连接层  
    fc2_weights = tf.get_variable("fc2_weights", [1024, num_class], initializer=tf.truncated_normal_initializer(stddev=0.01)) #神经元节点数1024, 分类节点10  
    fc2_biases = tf.get_variable("fc2_biases", [num_class], initializer=tf.constant_initializer(0.01))  
    fc2 = tf.matmul(fc1_dropout, fc2_weights) + fc2_biases  
      
    #第七层：输出层  
    # softmax  
    y_conv = tf.nn.softmax(fc2) 
    return y_conv



def loss(logits,label_batches):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=label_batches)
    cost = tf.reduce_mean(cross_entropy)
    return cost
 
    
def get_accuracy(logits,labels):
    acc = tf.nn.in_top_k(logits,labels,1)
    acc = tf.cast(acc,tf.float32)
    acc = tf.reduce_mean(acc)
    return acc


def training(loss,lr):
    train_op = tf.train.RMSPropOptimizer(lr,0.9).minimize(loss)
    return train_op


def run_training():
    data_dir = './use/'
    image,label = get_files(data_dir)
    image_batches,label_batches = get_batches(image,label,resize,resize,batch_size,20)
    p = model(image_batches)
    cost = loss(p,label_batches)
    train_op = training(cost,lr)
    acc = get_accuracy(p,label_batches)
    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess,coord = coord)
    
    try:
       for step in np.arange(steps):
           if coord.should_stop():
               break
           _,train_acc,train_loss = sess.run([train_op,acc,cost])
           print("train step:{} loss:{} accuracy:{}".format(step,train_loss,train_acc))
    except tf.errors.OutOfRangeError:
        print("Done!!!")
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
    

if __name__ == '__main__':
    run_training()