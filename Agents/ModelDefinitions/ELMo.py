from tensorflow.keras.layers import Dense, Activation, GRU, Bidirectional, Embedding
from tensorflow.keras import Sequential
from Layers.Attention import Attention
from Agents.DataObjects.TextData import TextData


class ELMo:
    model: Sequential
    data: TextData

    def __init__(self) -> None:
        self.model = Sequential()

    def initialize(self, data: TextData, size: int) -> None:
        self.data = data
        self.model.add(Embedding(
            input_dim=len(data.chars),
            input_shape=(self.data.maxLen, len(self.data.chars)),
            output_dim=size,
            input_length=data.maxLen
        ))

    def hiddenLayers(self, size: int) -> None:
        self.model.add(Attention(return_sequences=True))
        self.model.add(Bidirectional(GRU(size)))
        self.model.add(Dense(len(self.data.chars)))
        self.model.add(Activation('softmax'))

    def get(self) -> Sequential:
        return self.model
