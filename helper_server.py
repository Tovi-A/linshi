import tensorflow as tf


tf.app.flags.DEFINE_string("worker_hosts", "localhost:2222,localhost:2224", "worker hosts")
tf.app.flags.DEFINE_string("job_name", "worker", "'ps' or'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS

def main(_):
    worker_hosts = FLAGS.worker_hosts.split(",")
    # create cluster
    cluster = tf.train.ClusterSpec({"worker": worker_hosts})
    # create the server
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    server.join()

if __name__ == "__main__":
    # tf.app.run():先处理flag解析，再执行main函数
    tf.app.run()


# python helper_server.py --job_name=worker --task_index=0