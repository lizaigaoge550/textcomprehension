import tensorflow as tf
import numpy as np
from datetime import datetime

Len = 100000
batch_size = 5
epoch = 1
size = 1000

X = np.random.random_sample([Len, size])
Y = np.random.random_sample([Len, size])

#生成数据
def generator_batch_data():
    for i in range(0,Len,batch_size):
        if i + batch_size < Len:
            x = X[i:i+batch_size,:]
            y = Y[i:i+batch_size,:]
        else:
            x = np.concatenate(X[i:Len,:],X[0:i+batch_size - Len])
            y = np.concatenate(Y[i:Len,:],Y[0:i+batch_size - Len])
        yield (x,y)

def generator_data():
    for i in range(Len):
        yield (X[i],Y[i])


class GeneratorClass():
    def __init__(self, **params):
        self.x = params['x']
        self.y = params['y']
        self.result = self.x+self.y


class GeneratorClass_placehodler():
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[batch_size, size])
        self.y = tf.placeholder(dtype=tf.float32, shape=[batch_size, size])
        self.result = self.x + self.y


# ds = tf.data.Dataset.from_generator(generator_data,(tf.float32, tf.float32)).repeat(epoch).batch(batch_size)
# x,y = ds.make_one_shot_iterator().get_next()
print(datetime.now())

#generator = GeneratorClass(x=x,y=y)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     zong = epoch * np.ceil(Len//batch_size)
#     try:
#         while zong:
#             res = sess.run(generator.result)
#             print(res.shape)
#             zong -= 1
#     finally:
#         print(datetime.now())

generator = GeneratorClass_placehodler()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for batch in generator_batch_data():
        feed_dict = {}
        feed_dict[generator.x] = batch[0]
        feed_dict[generator.y] = batch[1]
        res = sess.run(generator.result, feed_dict)
        print(res.shape)
    print(datetime.now())