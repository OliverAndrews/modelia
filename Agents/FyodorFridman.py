from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout
from tensorflow.keras import Sequential
from Agents.DataObjects.TextData import TextData
from Utils.Silent import silent


@silent
class FyodorFridman:
    model: Sequential
    data: TextData

    def __init__(self) -> None:
        self.model = Sequential()

    def initialize(self, data: TextData) -> None:
        self.data = data
        self.model.add(LSTM(256, return_sequences=True, input_shape=(self.data.maxLen, len(self.data.chars))))

    def addFixedLayers(self) -> None:
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(256))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(len(self.data.chars)))
        self.model.add(Activation('softmax'))

    def get(self) -> Sequential:
        return self.model
