import tensorflow as tf

if __name__ == "__main__":
    with tf.device("/job:ps/task:0"):
        x = tf.Variable(tf.ones([2, 2]))
        y = tf.Variable(tf.ones([2, 2]))

    with tf.device("/job:worker/task:0"):
        z = tf.matmul(x, y) + x

    

    with tf.Session("grpc://192.168.1.104:2222") as sess:
        sess.run(tf.global_variables_initializer())
        val = sess.run(z)
        print(val)