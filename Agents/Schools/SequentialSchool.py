from tensorflow.keras import Sequential
from Agents.DataObjects.CompilerRules import CompilerRules
from Agents.DataObjects.TrainingData import TrainingData
from Agents.DataObjects.TrainingRules import TrainingRules
from Visualizers.Graph import Graph


class SequentialSchool:
    model: Sequential

    def __init__(self, model: Sequential) -> None:
        self.model = model

    def initialize(self, rules: CompilerRules) -> None:
        self.model.compile(
            loss=rules.lossFunction,
            optimizer=rules.optimizer)

    def fit(self, data: TrainingData, rules: TrainingRules) -> None:  # , rules: TrainingRules) -> None:
        if data.dtype is str:
            raise TypeError("String values are invalid training data for Sequential models")
        elif data.dtype is float:
            history = self.model.fit(
                data.floatTrainX,
                data.floatTrainY,
                epochs=rules.epochs,
                batch_size=rules.batchSize,
                verbose=rules.verbose)
            if rules.plotLoss:
                Graph.build([x for x, _ in enumerate(history.history['loss'])], history.history['loss'], title="Losses")
        elif data.dtype is int:
            raise Exception("Not implemented!")

    def get(self) -> Sequential:
        return self.model

    def __repr__(self) -> str:
        return f"Sequential School with model \n {self.model}"
