from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.layers import GRU, Dense, Bidirectional, Activation

from Agents.DataObjects.TextData import TextData
from Layers.Attention import Attention


class CoVe:
    model: Sequential
    data: TextData

    def __init__(self) -> None:
        self.model = Sequential()

    def initialize(self, data: TextData, size: int) -> None:
        self.data = data
        self.model.add(GRU(size, return_sequences=True, input_shape=(self.data.maxLen, len(self.data.chars))))

    def CoVeBlock(self, size: int, return_sequences=False) -> None:
        self.model.add(Attention(return_sequences=True))
        self.model.add(Bidirectional(GRU(size, return_sequences=return_sequences)))
        self.model.add(Bidirectional(GRU(size, return_sequences=return_sequences)))
        self.model.add(Bidirectional(GRU(size, return_sequences=False)))

    def OutputBlock(self) -> None:
        self.model.add(Dense(len(self.data.chars)))
        self.model.add(Activation('softmax'))

    def get(self) -> Sequential:
        return self.model
