from DataProcessing.Text.TextProcess import TextProcess
from Agents.DataObjects.TextData import TextData
from Agents.FyodorFridman import FyodorFridman
from Agents.DataObjects.TrainingRules import TrainingRules
from Agents.Schools.FyodorFridmanSchool import FyodorFridmanSchool
from Agents.ModelWrappers.TextSimulator import TextSimulator


if __name__ == '__main__':
    rulesTrain: TrainingRules = TrainingRules(epochs=10000, plotLoss=False, verbose=1, batchSize=200)
    data: TextData = TextProcess.vectorizeTextFile("Data/Text/lex", TextData())

    # Configuring model
    lex: FyodorFridman = FyodorFridman()
    lex.initialize(data)
    lex.addFixedLayers()

    # Configuring school
    school: FyodorFridmanSchool = FyodorFridmanSchool(lex.get(), data)
    school.train(rulesTrain, learningRate=0.001)

    # Predict some text from Notes From Underground
    [TextSimulator.generate(data, school.get(), diversity=0.1) for _ in range(50)]
