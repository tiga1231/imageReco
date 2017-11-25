import os, json

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def makeW(shape, name=None):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init, name=name)

def makeB(shape, name=None):
    init = tf.constant(0.1,shape=shape)
    return tf.Variable(init, name=name)

def conv(x, W, name=None):
    '''four dimensions:
    input index
    input x-position
    input y-position
    feature index
    '''
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name=name)

def pool(x, name=None):
    #return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)
    return tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def merge(l):
    res = l[0]
    for i in range(1,len(l)):
        batch = l[i]
        res['data'] = np.concatenate([res['data'], batch['data']], axis=0)
        res['labels'] += batch['labels']
    return res

def nextBatch(u, n):
    import random
    sample = random.sample(xrange(len(labels)), n)
    d = np.array([data[i] for i in sample])
    l = np.array([labels[i] for i in sample])
    return [d, l]

def oneHot(l):
    m = np.max(l)
    res = np.zeros([len(l),m+1])
    l = np.array(l)
    res[xrange(len(l)), l] = 1
    return res



#-------- data fetch ---------------
labelCategories = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
u = [unpickle('cifar-10-batches-py/data_batch_'+str(i+1)) for i in range(5)]
u = merge(u)
data = u['data']
labels = u['labels']
labels = oneHot(labels)

v = unpickle('cifar-10-batches-py/test_batch')


#----------- in/out --------------
x = tf.placeholder(tf.float32, shape=[None, 32*32*3], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')


img = tf.transpose( tf.reshape(x, [-1,3,32,32]),  [0, 2, 3, 1], name='img') / 255.0

#--------- 1st layer -------------
with tf.variable_scope('conv1'):
    w1 = makeW([5,5,3,32], name='w1')
    b1 = makeB([32], name='b1')

    h1 = tf.nn.relu(conv(img, w1 ) + b1)
    p1 = pool(h1)


#--------- 2nd layer -------------
with tf.variable_scope('conv2'):
    w2 = makeW([5,5,32,64], name='w2')
    b2 = makeB([64], name='b2')

    h2 = tf.nn.relu(conv(p1, w2) + b2)
    p2 = pool(h2)


#--------- fully connected (fc) layer -------------
with tf.variable_scope('fc1'):
    wfc1 = makeW([8*8*64, 1024], name='w_fc1')
    bfc1 = makeB([1024], name='b_fc1')
    flat = tf.reshape(p2, [-1,8*8*64])
    hfc1 = tf.nn.relu(tf.matmul(flat, wfc1) + bfc1)


#--------- dropout -------------
with tf.variable_scope('dropout'):
    prob = tf.placeholder(tf.float32)
    drop = tf.nn.dropout(hfc1, prob)


#--------- readout -------------
with tf.variable_scope('fc2'):
    wfc2 = makeW([1024, 10], name='w_fc2')
    bfc2 = makeB([10], name='b_fc2')
    y = tf.matmul(drop, wfc2) + bfc2

#--------- optimizer & eval ----------------

with tf.variable_scope('cross_entropy'):
    ce = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y), name='cross_entropy')

with tf.variable_scope('accuracy'):
    correct = tf.equal(tf.argmax(y_,1,name='label'), tf.argmax(y,1, name='pred_label'), name='correct')
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

train = tf.train.AdamOptimizer(1e-4).minimize(ce)



#---------- gradients for GradCAM --------

instance = tf.placeholder(tf.int32)
g_h1 = [tf.gradients(y[instance, i], h1) for i in range(10)]
g_p1 = [tf.gradients(y[instance, i], p1) for i in range(10)]
g_h2 = [tf.gradients(y[instance, i], h2) for i in range(10)]
g_p2 = [tf.gradients(y[instance, i], p2) for i in range(10)]


#----------- session --------------------
saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    saver.restore(sess, "./save/model.ckpt")

    for i in range(1):
        batch = nextBatch(u, 100)
        train.run(feed_dict={x:batch[0], y_:batch[1], prob:0.5})

        if i%100 == 99:
            ai = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], prob:1.0})
            batchTest = nextBatch(v, 100)
            bi = accuracy.eval(feed_dict={x:batchTest[0], y_:batchTest[1], prob:1.0})
            print i, ai, bi
            saver.save(sess, "./save/model.ckpt")
        elif i%50==0:
            print i
    
    # ------ gradCAM imshow ----------
    def cal(im, g):
        res = np.zeros(im.shape[:2])
        #print len(g), g
        for i in range(len(g)):
            #res += g[:,:,i] * im[:,:,i]
            #res += g[:,:,i].mean() * im[:,:,i]
            res += np.median(g[:,:,i]) * im[:,:,i]
            
            #z = g[:,:,i]#*(g[:,:,i]>0)
            #softmax = np.exp(z) / np.sum(z)
            #res += softmax * im[:,:,i]
            
            #res += im[:,:,i]
        return res


    colors = [(0, 0, 1), (0.9, 0.9, 0.9), (1, 0, 0)]
    colors = [(0, 0, 1), (0, 0, 0), (1, 1, 0)]
    #colors = [(0.8, 0.8, 0.8), (1, 0, 0)]

    cm = LinearSegmentedColormap.from_list('aa', colors)
    
    for _ in range(10):#random instances
        batch = nextBatch(u, 1)

        pred = sess.run(y, feed_dict={x:batch[0], prob:1.0})
        pred = pred.argmax(axis=1)
        truth = batch[1].argmax(axis=1)
    
        plt.figure(1, figsize=[14,8])

        plt.subplot(5,3,2)
        originalImg = batch[0].reshape([3,32,32])
        originalImg = np.swapaxes(originalImg, 0,1)
        originalImg = np.swapaxes(originalImg, 1,2)
        plt.imshow(originalImg)
        plt.title(labelCategories[pred[0]] + ' / ' + labelCategories[truth[0]])
        plt.axis('off')

        imgss = []
        vmax = []
        for j,g,dx in zip(range(4), [g_h1, g_p1, g_h2, g_p2], [h1, p1, h2, p2]):

            gg ,im = sess.run([g, dx], feed_dict={x:batch[0], y_:batch[1], prob:1.0, instance:0})
            imgs = []

            img = im[0]

            #--- activation map of all categories ------
            '''
            plt.figure(j+2)
            for i in range(img.shape[2]):
                plt.subplot(8,8,i+1)
                plt.imshow(img[:,:,i])
                plt.axis('off')
            plt.savefig('a_' + str(j)+'.png')
            '''

            #--- gradient map of all categories ------
            for c in range(len(gg)):#category class
                g = gg[c][0]
                g = g[0,:,:,:]

                # calculate Grad-CAM
                res = cal(img, g)
                imgs.append(res)
            vmax.append(np.max(np.abs(imgs)))
            imgss.append(imgs)

            
        plt.figure(1)

        for j in range(4):
            imgs = imgss[j]

            maxsum = np.max([imgs[c][imgs[c]>0].sum() for c in range(10)])

            for c in range(10):#category class
                plt.subplot(5,10,10*(j+1)+c+1)

                imgvmax = np.abs(imgs[c]).max()
                #plt.imshow(originalImg, extent=(0,1,0,1), alpha=1)

                #plt.imshow(imgs[c], vmax=vmax[j], vmin=-vmax[j], cmap=cm, extent=(0,1,0,1), alpha=1)
                plt.imshow(imgs[c], vmax=imgvmax, vmin=-imgvmax, cmap=cm, extent=(0,1,0,1), alpha=1)
                
                m,n = imgs[c].shape
                X, Y = np.meshgrid(np.linspace(0,1,m), np.linspace(1,0,n))
                levels = np.linspace(-imgvmax, imgvmax, 10)
                #levels = np.linspace(-vmax[j], vmax[j], 20)
                #plt.contourf(X, Y, imgs[c], levels, cmap=cm)

                act = imgs[c][imgs[c]>0].sum()
                s = '%.2f' % act
                if act == maxsum:
                    color = 'red'
                else:
                    color = 'black'
                if j==0:
                    plt.title(labelCategories[c] + '\n' + s, color=color)
                else:
                    plt.title(s, color=color)
                plt.axis('equal')
                plt.axis('off')

        plt.savefig(str(_) + '.png', dpi=150)
        #plt.show()

        
        
    

    #----- training set accuracy (SLOW)--------
    #print 'train accuracy:', sess.run(accuracy, feed_dict={x:data, y_:labels, prob:1.0})


    #----- test set accuracy ------------------
    
    '''
    print 'test accuracy:', sess.run(accuracy, feed_dict={x:data, y_:labels, prob:1.0})
    '''



    
    #------------show pred instances -------------------
    '''
    batch = nextBatch(u, 100)#[data[:100], labels[:100]]
    image,pred = sess.run([img,y], feed_dict={x:batch[0], prob:1.0})
    pred = pred.argmax(axis=1)
    truth = batch[1].argmax(axis=1)
    print image.shape
    plt.figure(figsize=[10,15])
    for i in range(100):
        plt.subplot(10,10,i+1)
        plt.imshow(image[i][:,:,[0,1,2]])
        plt.xticks([])
        plt.yticks([])
        if pred[i] == truth[i]:
            plt.title(labelCategories[truth[i]], fontsize=8)
        else:
            plt.title(labelCategories[pred[i]] + '/' + labelCategories[truth[i]], color='red', fontsize=8)
    plt.subplots_adjust(left=0.05, bottom=0.05, 
                    right=0.95, top=0.95,
                    wspace=0.0, hspace=0.2)
    #plt.show()
    plt.savefig('test.png')
    '''


#----------------------------
writer.close()














