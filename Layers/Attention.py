from tensorflow.keras.backend import tanh, softmax, dot, sum
from tensorflow.keras.layers import Layer


class Attention(Layer):
    """
                ATTENTION LAYER
    =============================================
            Taken and modified from:
    https://stackoverflow.com/a/62949137/10614851
    """

    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")

        super(Attention, self).build(input_shape)


    def call(self, x):
        e = tanh(dot(x, self.W) + self.b)
        a: object = softmax(e, axis=1)
        output = x * a

        if self.return_sequences:
            return output

        return sum(output, axis=1)
