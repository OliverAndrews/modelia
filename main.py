import Utils.CompilerInformation  # Silence Tensorflow compiler info
from DataProcessing.Text.TextProcess import TextProcess
from Agents.DataObjects.TextData import TextData
from Agents.ModelDefinitions.CoVe import CoVe
from Agents.DataObjects.TrainingRules import TrainingRules
from Agents.Schools.FyodorFridmanSchool import FyodorFridmanSchool

if __name__ == '__main__':
    rulesTrain: TrainingRules = TrainingRules(
        epochs=100,
        plotLoss=False,
        verbose=1,
        batchSize=36,
        floatContainerOne=0.5,  # Learning Rate
        floatContainerTwo=0.0  # This is momentum
    )
    data: TextData = TextProcess.vectorizeTextFile("Data/Text/notesfromunderground", TextData())

    # Configuring model
    model: CoVe = CoVe()
    model.initialize(data, size=200)
    model.CoVeBlock(200, return_sequences=True)
    model.OutputBlock()

    # Configuring school
    school: FyodorFridmanSchool = FyodorFridmanSchool(model.get(), data)
    school.train(rulesTrain)
    school.get().save('./Models')
