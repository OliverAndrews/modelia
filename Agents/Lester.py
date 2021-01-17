from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from Utils.Silent import silent


@silent
class Lester:
    model: Sequential

    def __init__(self, inputDim: int, outputDim: int) -> None:
        self.model = Sequential()
        self.model.add(Embedding(input_dim=inputDim, output_dim=outputDim))

    def addLSTMRange(self, start: int, end: int = 1, step: int = 1) -> None:
        for i in reversed(range(start, end, step)):
            self.model.add(LSTM(i))

    def addSingleLSTM(self, units: int) -> None:
        self.model.add(LSTM(units))

    def addSingleDense(self, units: int) -> None:
        self.model.add(Dense(units))

    def __repr__(self) -> str:
        return f"{self.model.summary()}"

    def get(self) -> Sequential:
        return self.model
