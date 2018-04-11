import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle

df = pd.read_csv("/home/user/Downloads/portugese bank/bank-full-encoded.csv", sep=";")
df1 = df[df.y == 1]
df1 = shuffle(df1)
df0 = df[df.y == 0]
df0 = shuffle(df0)

df = pd.DataFrame()
df = pd.concat([df, df1[:5000]], axis=0)
df = pd.concat([df, df0[:5000]], axis=0)
df = shuffle(df)

Y = pd.DataFrame(df["y"])
X = df.drop("y", axis=1)

print("generated data...")

X_data = tf.placeholder("float", shape=[None, 26])
Y_data = tf.placeholder("float", shape=[None, 1])


W_1 = tf.Variable(tf.random_normal(shape=[26, 100]))
b_1 = tf.Variable(tf.random_normal(shape=[100]))
a1 = tf.sigmoid(tf.add(tf.matmul(X_data, W_1), b_1))

W_2 = tf.Variable(tf.random_normal(shape=[100, 75]))
b_2 = tf.Variable(tf.random_normal(shape=[75]))
a2 = tf.sigmoid(tf.add(tf.matmul(a1, W_2), b_2))

W_out = tf.Variable(tf.random_normal(shape=[75, 1]))
b_out = tf.Variable(tf.random_normal(shape=[1]))
logit = tf.add(tf.matmul(a2, W_out), b_out)

print(Y_data)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_data, logits=logit))
optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epochs = 400
    batch_size = 25
    batches = int(len(X) / batch_size)

    for _ in range(epochs):
        for i in range(batches):
            sess.run(optimizer, feed_dict={X_data: X[(i*batch_size):(i+1)*batch_size],
                                           Y_data: Y[(i*batch_size):(i+1)*batch_size]})
        print("cost after 1 epoch is", sess.run(cost, feed_dict={X_data:X, Y_data: Y}))
    sess.close()
