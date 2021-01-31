from DataProcessing.Text.TextProcess import TextProcess
from Agents.DataObjects.TextData import TextData
from Agents.ModelDefinitions.FyodorFridman import FyodorFridman
from Agents.DataObjects.TrainingRules import TrainingRules
from Agents.Schools.FyodorFridmanSchool import FyodorFridmanSchool
from Agents.ModelWrappers.TextSimulator import TextSimulator

if __name__ == '__main__':
    rulesTrain: TrainingRules = TrainingRules(
        epochs=100,
        plotLoss=False,
        verbose=1,
        batchSize=36,
        floatContainerOne=0.01,  # Learning Rate
        floatContainerTwo=0.0  # This is momentum
    )
    data: TextData = TextProcess.vectorizeTextFile("Data/Text/notesfromunderground", TextData())

    # Configuring model
    lex: FyodorFridman = FyodorFridman()
    lex.initialize(data, gru=True)
    lex.addFixedGRULayers()

    # Configuring school
    school: FyodorFridmanSchool = FyodorFridmanSchool(lex.get(), data)
    school.train(rulesTrain)

    # Predict some text from Notes From Underground
    [TextSimulator.generate(data, school.get(), diversity=0.01) for _ in range(50)]
