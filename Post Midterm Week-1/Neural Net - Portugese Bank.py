import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

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

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.3, random_state= 30)

print("generated data...")

X_data = tf.placeholder("float", shape=[None, 26])
Y_data = tf.placeholder("float", shape=[None, 1])


W_1 = tf.Variable(tf.random_normal(shape=[26, 200]))
b_1 = tf.Variable(tf.random_normal(shape=[200]))
a1 = tf.sigmoid(tf.add(tf.matmul(X_data, W_1), b_1))

W_2 = tf.Variable(tf.random_normal(shape=[100, 175]))
b_2 = tf.Variable(tf.random_normal(shape=[175]))
a2 = tf.sigmoid(tf.add(tf.matmul(a1, W_2), b_2))

W_out = tf.Variable(tf.random_normal(shape=[175, 1]))
b_out = tf.Variable(tf.random_normal(shape=[1]))
logit = tf.add(tf.matmul(a2, W_out), b_out)
answer = tf.round(tf.sigmoid(logit))

print(Y_data)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_data, logits=logit))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_data, answer), "float32"))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epochs = 200
    batch_size = 25
    batches = int(len(X) / batch_size)

    for j in range(epochs):
        for i in range(batches):
            sess.run(optimizer, feed_dict={X_data: X_train[(i*batch_size):(i+1)*batch_size],
                                           Y_data: Y_train[(i*batch_size):(i+1)*batch_size]})
        print("testing cost after", j," epoch is", sess.run(cost, feed_dict={X_data:X_test, Y_data: Y_test}))
        print("accuracy after", j," epoch is", sess.run(accuracy, feed_dict={X_data:X_test, Y_data: Y_test}))

    sess.close()
