from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential
from Agents.ModelWrappers.TextSimulator import TextSimulator
from Agents.DataObjects.TextData import TextData


class TextCallback(Callback):
    data: TextData
    diversity: float

    def __init__(self, data: TextData, diversity: float):
        super().__init__()
        self.data = data
        self.diversity = diversity

    def on_epoch_end(self, batch, logs=None):
        model: Sequential = self.model
        print()
        TextSimulator.generate(model=model, data=self.data, diversity=self.diversity)

