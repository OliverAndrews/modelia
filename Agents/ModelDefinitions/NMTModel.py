from tensorflow_addons.utils.types import Activation
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.layers import GRU, LSTM, Dropout, Dense, Bidirectional

from Agents.DataObjects.TextData import TextData
from Layers.Attention import Attention
from Utils.Silent import silent


@silent
class NMTModel:
    model: Sequential
    data: TextData

    def __init__(self) -> None:
        self.model = Sequential()

    def initialize(self, data: TextData) -> None:
        self.data = data
        self.model.add(GRU(200, return_sequences=True, input_shape=(self.data.maxLen, len(self.data.chars))))

    def NMTBlock(self, size: int, return_sequences=False) -> None:
        self.model.add(Attention(return_sequences=True))
        self.model.add(Bidirectional(GRU(size, return_sequences=return_sequences)))

    def OutputBlock(self) -> None:
        self.model.add(Dense(len(self.data.chars)))
        self.model.add(Activation('softmax'))

    def get(self) -> Sequential:
        return self.model
