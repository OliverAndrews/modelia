from tensorflow.keras import Sequential
from Agents.DataObjects.TextData import TextData
from Agents.DataObjects.TrainingRules import TrainingRules
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping

class FyodorFridmanSchool:
    model: Sequential
    data: TextData

    def __init__(self, model: Sequential, data: TextData):
        self.model = model
        self.data = data

    def train(self, rules: TrainingRules, learningRate: float) -> None:
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=SGD(
                learning_rate=rules.floatContainerOne,
                momentum=rules.floatContainerTwo,
                nesterov=False))
        self.model.fit(
            self.data.testX,
            self.data.testY,
            verbose=rules.verbose,
            batch_size=rules.batchSize,
            epochs=rules.epochs,
            callbacks=[EarlyStopping(monitor='loss', patience=3)])

    def get(self) -> Sequential:
        return self.model
