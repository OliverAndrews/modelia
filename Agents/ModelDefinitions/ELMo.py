from tensorflow.keras.layers import Dense, Activation, GRU, Bidirectional, Embedding
from tensorflow.keras import Sequential
from Layers.Attention import Attention
from Agents.DataObjects.TextData import TextData
from Utils.Silent import silent


@silent
class ELMo:
    model: Sequential
    data: TextData

    def __init__(self) -> None:
        self.model = Sequential()

    def initialize(self, data: TextData, size: int) -> None:
        self.data = data
        # GRU(200, return_sequences=True, input_shape=(self.data.maxLen, len(self.data.chars)))
        self.model.add(Embedding(
            input_dim=10,
            output_dim=2
        ))  # Wrong! Needs a fix.

    def addFixedGRULayers(self, size: int) -> None:
        self.model.add(Attention(return_sequences=True))
        self.model.add(Bidirectional(GRU(size)))
        self.model.add(Dense(len(self.data.chars)))
        self.model.add(Activation('softmax'))

    def get(self) -> Sequential:
        return self.model
