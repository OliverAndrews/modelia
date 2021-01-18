from tensorflow.keras import Sequential
from typing import List

from Agents.DataObjects.TrainingData import TrainingData


class SequentialSimulator:

    model: Sequential

    def __init__(self, model):
        self.model = model

    def predictSinsoid(self, data: TrainingData) -> None:
        pass