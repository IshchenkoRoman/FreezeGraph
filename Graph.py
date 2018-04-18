import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from time import process_time
import time

from subprocess import call

def load_graph(graph_file, use_xla=False):
    config = tf.ConfigProto()
    if use_xla:
        jit_level = tf.OptimizerOptions.ON_1
        config.graph_options.optimizer_options.global_jit_level = jit_level

    with tf.Session(graph=tf.Graph(), config=config) as sess:
        gd = tf.GraphDef()
        with tf.gfile.Open(graph_file, 'rb') as f:
            data = f.read()
            gd.ParseFromString(data)
        tf.import_graph_def(gd, name='')
        ops = sess.graph.get_operations()
        n_ops = len(ops)
        return sess.graph, ops

def printMsg(msg, arg):
    print("\n================\n" + msg + "{0}\n================\n".format(arg))

# def freeze_graph(sess):
#
#     graph_def = tf.get_default_graph().as_graph_def()
#     output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['y_CNN'])
#
#     with tf.gfile.GFile("Frozen_CNN.pb", "wb") as f:
#         f.write(output_graph_def.SerializeToString())

def load_meta_graph(mnist):

    """
    Load .meta graph
    :param mnist: mnist- tensorflow data set of labeled numbers
    :return:
    """
    with tf.Session() as sess:

        new_saver = tf.train.import_meta_graph("./CNN.meta")
        new_saver.restore(sess, "./CNN")

        graph = tf.get_default_graph()
        batch = mnist.train.next_batch(50)
        x = graph.get_tensor_by_name("x:0")
        y_ = graph.get_tensor_by_name("y_:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        init_op = tf.global_variables_initializer()
        y_CNN = graph.get_tensor_by_name("y_CNN:0")

        with tf.Session(graph=graph) as sess:
            sess.run(init_op)
            start = time.time()
            print(sess.run(y_CNN, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1}))
            end = time.time()

            print("Meta = {0:.6f}".format(end - start))

def load_pb_graph(mnist):

    """
    Load freezed graph in format .pb
    :param mnist: mnist- tensorflow data set of labeled numbers
    :return: None
    """

    with tf.gfile.GFile("frozen_graph.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    batch = mnist.train.next_batch(50)
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            name="prefix"
        )

    x = graph.get_tensor_by_name("prefix/x:0")
    kp = graph.get_tensor_by_name("prefix/keep_prob:0")
    y_CNN = graph.get_tensor_by_name("prefix/y_CNN:0")

    with tf.Session(graph=graph) as sess:
        #[tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        start = time.time()
        print(sess.run(y_CNN, feed_dict={x: batch[0], kp: 1}))
        end = time.time()

        print("Frozen = {0:.6f}".format(end - start))

def ConvNet(mnist):


    """
    ConvNet(mnist)
        Simple Convolution NNet with 4 layers (3 hidden)
        All thensors must have "name=" for get access by function get_tensor_by_name()/ get_operation_by_name()
    :param mnist: mnist- tensorflow data set of labeled numbers
    :return: None
    """
    # Shape of image and count of numbers from 0 to 9 (see mnist)
    width = 28
    height = 28
    flat = width * height
    class_out = 10

    x = tf.placeholder(tf.float32, shape=[None, flat], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, class_out], name="y_")

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    printMsg("x_image = ", x_image.get_shape())

    # First layer
    W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1), name="W1")
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]), name="b1")
    convolve1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1],
                             padding="SAME", name="conv1") + b_conv1

    h_conv1 = tf.nn.relu(convolve1)
    conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                            padding="SAME", name="maxpool1") # maxpool 2X2
    printMsg("Conv1 shape = ", conv1.get_shape())

    # Second layer
    W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), name="W2")
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]), name="b2")

    convolve2 = tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1],
                             padding="SAME", name="conv2") + b_conv2

    h_conv2 = tf.nn.relu(convolve2)
    conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                            padding="SAME", name="maxpool2")
    printMsg("Conv2 shape = ", conv2.get_shape())

    # Third layer
    layer2_matrix = tf.reshape(conv2, [-1, 7*7*64])
    W_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1), name="Wfc1")
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]), name="bfc1")

    fcl = tf.matmul(layer2_matrix, W_fc1) + b_fc1
    h_fc1 = tf.nn.relu(fcl)

    printMsg("Full connected shape = ", h_fc1.get_shape())

    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    layer_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = tf.Variable(tf.truncated_normal([1024, 10]), name="Wfc2")
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]), name="bfc2")

    fc = tf.matmul(layer_drop, W_fc2) + b_fc2

    y_CNN = tf.nn.softmax(fc, name="y_CNN")

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]))

    train_step = tf.train.AdamOptimizer(1e-4, name="optimizer").minimize(cross_entropy, name="output")

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)

        for i in range(1100):
            batch = mnist.train.next_batch(50)
            res = sess.run(y_CNN, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        tf.train.write_graph(tf.Session().graph_def, "./", "model.pb", as_text=False)
        saver.save(sess=sess, save_path="./CNN")

        sess.close()


def test_speed(mnist):

    with tf.gfile.GFile("frozen_graph.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    batch = mnist.train.next_batch(50)
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            name="prefix"
        )

    x = graph.get_tensor_by_name("prefix/x:0")
    kp = graph.get_tensor_by_name("prefix/keep_prob:0")
    y_CNN = graph.get_tensor_by_name("prefix/y_CNN:0")

    with tf.Session(graph=graph) as sess:
        #[tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        start = time.time()
        sess.run(y_CNN, feed_dict={x: batch[0], kp: 1})
        end = time.time()

        print("Frozen = {0:.6f}".format(end - start))

    with tf.Session(graph=graph) as sess:
        #[tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        start = time.time()
        sess.run(y_CNN, feed_dict={x: batch[0], kp: 1})
        end = time.time()

        print("Frozen = {0:.6f}".format(end - start))

    with tf.gfile.GFile("Optimize_graph.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    batch = mnist.train.next_batch(50)
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            name="prefix"
        )

    x = graph.get_tensor_by_name("prefix/x:0")
    kp = graph.get_tensor_by_name("prefix/keep_prob:0")
    y_CNN = graph.get_tensor_by_name("prefix/y_CNN:0")

    with tf.Session(graph=graph) as sess:
        #[tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        start = time.time()
        sess.run(y_CNN, feed_dict={x: batch[0], kp: 1})
        end = time.time()

        print("Optimize_graph = {0:.6f}".format(end - start))


def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    ConvNet(mnist)
    load_meta_graph(mnist)
    call(["sh", "StartFreeze.sh"])
    load_pb_graph(mnist)
    call(["sh", "OptimizeFozen.sh"])

    _, base_ops = load_graph("model.pb")
    print("Len ops for base model = {0}".format(len(base_ops)))
    _, base_ops = load_graph("frozen_graph.pb")
    print("Len ops for frozen model = {0}".format(len(base_ops)))
    _, base_ops = load_graph("Optimize_graph.pb")
    print("Len ops for optimize model = {0}".format(len(base_ops)))

    test_speed(mnist)


if __name__=="__main__":
    main()
