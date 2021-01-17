import tensorflow as tf


class Mute:
    @staticmethod
    def muteTensorflow():
        tf.get_logger().setLevel('ERROR')
