from Agents.DataObjects.CompilerRules import CompilerRules
from Agents.Lester import Lester
from Utils.Preprocessing import Preprocessing
from Agents.DataObjects.TrainingData import TrainingData
from Generators.SineWave import SineWave
from numpy import ndarray
from Agents.Schools.SequentialSchool import SequentialSchool
from Agents.DataObjects.TrainingRules import TrainingRules

if __name__ == '__main__':

    # Synthesizing Data
    wave: SineWave = SineWave(5000)
    wave.initialize(5000)
    x: ndarray = wave.series()
    y: ndarray = wave.generate()

    # Building Data Objects
    data: TrainingData = TrainingData(floatAllX=x, floatAllY=y, dtype=float)
    rulesTrain: TrainingRules = TrainingRules(epochs=20, plotLoss=True, verbose=1, batchSize=20)
    rulesCompiler: CompilerRules = CompilerRules(lossFunction="mse", optimizer="adam")

    # Getting Data Ready
    data = Preprocessing.trainTestSplit(ratio=0.3, data=data)
    data = Preprocessing.reshapeLSTM(data=data, maxSampleSize=200, features=1)

    # Building Models
    lester: Lester = Lester()

    # TODO: Setze diese in das Datenobjekt
    lester.initialize(inputShape=(data.floatTrainX.shape[1], 1))
    lester.addSingleDense(1)

    # Training Models
    school: SequentialSchool = SequentialSchool(model=lester.get())
    school.initialize(rulesCompiler)
    school.fit(data, rulesTrain)
