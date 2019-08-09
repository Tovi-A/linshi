from  __future__  import division, print_function, absolute_import

import tensorflow as tf

# Get data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# Some numbers
batch_size = 128
display_step = 10
num_input = 784
num_classes = 10

def conv_layer(inputs, channels_in, channels_out, strides=1):       
        
        # Create variables
        w = tf.Variable(tf.random_normal([3, 3, channels_in, channels_out]))
        b = tf.Variable(tf.random_normal([channels_out]))
        
        # We can double check the device that this variable was placed on
        print("w:", w.device) 
        print("b:", b.device)
        
        # "SAME"在不足时候会填充，而"VALID"会舍弃
        # Define Ops
        x = tf.nn.conv2d(inputs, w, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        
        # Non-linear activation
        return tf.nn.relu(x)

    
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# Create model
def CNN(x):
    
    with tf.device("/job:worker/task:0"): # <----------- Put first half of network on device 0

        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer
        conv1 = conv_layer(x, 1, 32, strides=1)
        pool1 = maxpool2d(conv1)

        # Convolution Layer
        conv2=conv_layer(pool1, 32, 64, strides=1)
        pool2=maxpool2d(conv2)

    with tf.device("/job:worker/task:1"):  # <----------- Put second half of network on device 1
        # Fully connected layer
        fc1 = tf.reshape(pool2, [-1, 7*7*64])
        w1 = tf.Variable(tf.random_normal([7*7*64, 1024]))
        b1 = tf.Variable(tf.random_normal([1024]))
        fc1 = tf.add(tf.matmul(fc1,w1),b1)
        fc1 = tf.nn.relu(fc1)

        # Output layer
        w2 = tf.Variable(tf.random_normal([1024, num_classes]))
        b2 = tf.Variable(tf.random_normal([num_classes]))
        out = tf.add(tf.matmul(fc1,w2),b2)
        
        # Check devices for good measure
        print("w1:", w1.device)
        print("b1:", b1.device)
        print("w2", w2.device)
        print("b2", b2.device)

    return out


# Construct model
with tf.device("/job:worker/task:1"):
    X = tf.placeholder(tf.float32, [None, num_input]) # Input images feedable
    Y = tf.placeholder(tf.float32, [None, num_classes]) # Ground truth feedable
    print("X:", X.device)
    print("Y:", Y.device)
    
logits = CNN(X) # Unscaled probabilities

with tf.device("/job:worker/task:1"):
    
    prediction = tf.nn.softmax(logits) # Class-wise probabilities
    
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
    # tf.argmax(vector, 1):返回vector最大值的索引号
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

# Start training
with tf.Session("grpc://192.168.1.104:2222") as sess:  # <----- IMPORTANT: Pass the server target to the session definition

    # Run the initializer
    sess.run(init)

    for step in range(100):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y : batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))

    # Get test set accuracy
    print("Testing Accuracy:",sess.run(accuracy, feed_dict={X: mnist.test.images[:256],Y: mnist.test.labels[:256]}))