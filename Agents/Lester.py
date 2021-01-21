from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras import Sequential
from Utils.Silent import silent
from typing import Tuple


@silent
class Lester:
    model: Sequential

    def __init__(self) -> None:
        self.model = Sequential()

    def initialize(self, inputShape: Tuple[int, int], returnSequence: bool = False) -> None:
        self.model.add(LSTM(10, return_sequences=returnSequence, input_shape=inputShape))

    def addLSTMRange(self, start: int, end: int = 1, step: int = 1) -> None:
        for i in reversed(range(start, end, step)):
            self.model.add(LSTM(i))

    def addDenseRange(self, start: int, end: int = 1, step: int = 1, activation: str ="tanh") -> None:
        for i in reversed(range(start, end, step)):
            self.model.add(Dense(i, activation=activation))


    def addSingleLSTM(self, units: int, returnSequence: bool = False) -> None:
        self.model.add(LSTM(units, activation="tanh", return_sequences=returnSequence))

    def addSingleDense(self, units: int) -> None:
        self.model.add(Dense(units, activation="tanh"))

    def __repr__(self) -> str:
        return f"{self.model.summary()}"

    def get(self) -> Sequential:
        return self.model
