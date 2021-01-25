from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout, GRU, Attention
from tensorflow.keras import Sequential
from Agents.DataObjects.TextData import TextData
from Utils.Silent import silent


@silent
class FyodorFridman:
    model: Sequential
    data: TextData

    def __init__(self) -> None:
        self.model = Sequential()

    def initialize(self, data: TextData, gru: bool = False) -> None:
        self.data = data
        if not gru:
            self.model.add(LSTM(512, return_sequences=True, input_shape=(self.data.maxLen, len(self.data.chars))))
        else:
            self.model.add(GRU(512, return_sequences=True, input_shape=(self.data.maxLen, len(self.data.chars))))

    def addFixedLayers(self) -> None:

        self.model.add(Dropout(0.5))
        self.model.add(LSTM(256))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(self.data.chars)))
        self.model.add(Activation('softmax'))

    # Attention Layer, which I am not sure how to use.
    @staticmethod
    def addAttentionLayer(useScale: bool = False) -> Attention:
        return Attention(use_scale=useScale)

    def addFixedGRULayers(self) -> None:
        self.model.add(Dropout(0.1))
        self.model.add(GRU(512, return_sequences=True))
        self.model.add(Dropout(0.1))
        self.model.add(GRU(512, return_sequences=True))
        self.model.add(Dropout(0.1))
        self.model.add(GRU(512))
        self.model.add(Dropout(0.1))  # Attilastic
        self.model.add(Dense(len(self.data.chars)))
        self.model.add(Activation('softmax'))

    def get(self) -> Sequential:
        return self.model
