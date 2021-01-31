import Utils.CompilerInformation  # Silence Tensorflow compiler info
from DataProcessing.Text.TextProcess import TextProcess
from Agents.DataObjects.TextData import TextData
from Agents.ModelDefinitions.ELMo import ELMo
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
    model: ELMo = ELMo()
    model.initialize(data, size=200)
    model.addFixedGRULayers(200)

    # Configuring school
    school: FyodorFridmanSchool = FyodorFridmanSchool(model.get(), data)
    school.train(rulesTrain)
