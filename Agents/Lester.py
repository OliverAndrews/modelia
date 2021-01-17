from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras import Sequential
from Utils.Silent import silent


@silent
class Lester:
    model: Sequential

    def __init__(self) -> None:
        self.model = Sequential()
        self.model.add(LSTM(30, return_sequences=True, input_shape=(None, 5)))

    def addLSTMRange(self, start: int, end: int = 1, step: int = 1) -> None:
        for i in reversed(range(start, end, step)):
            self.model.add(LSTM(i))

    def addSingleLSTM(self, units: int) -> None:
        self.model.add(LSTM(units, activation="tanh"))

    def addSingleDense(self, units: int) -> None:
        self.model.add(Dense(units, activation="tanh"))

    def __repr__(self) -> str:
        return f"{self.model.summary()}"

    def get(self) -> Sequential:
        return self.model
