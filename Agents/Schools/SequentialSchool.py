from tensorflow.keras import Sequential
from Agents.DataObjects.CompilerRules import CompilerRules
from Agents.DataObjects.TrainingData import TrainingData
from Agents.DataObjects.TrainingRules import TrainingRules


class SequentialSchool:
    model: Sequential

    def __init__(self, model: Sequential) -> None:
        self.model = model

    def initialize(self, rules: CompilerRules) -> None:
        self.model.compile(
            loss=rules.lossFunction,
            optimizer=rules.optimizerString,
            metrics=rules.metrics)

    def fit(self, data: TrainingData, rules: TrainingRules) -> None:
        if data.dtype is str:
            raise TypeError("String values are invalid training data for Sequential models")
        elif data.dtype is float:
            self.model.fit(
                data.floatTrainX,
                data.floatTrainY,
                validation_data=(data.floatTestX, data.floatTestY),
                batch_size=rules.batchSize,
                epochs=rules.epochs)
        elif data.dtype is int:
            self.model.fit(
                data.intTrainX,
                data.intTrainY,
                validation_data=(data.intTestX, data.intTestY),
                batch_size=rules.batchSize,
                epochs=rules.epochs)

    def __repr__(self) -> str:
        return f"Sequential School with model \n {self.model}"
