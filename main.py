from Agents.DataObjects.CompilerRules import CompilerRules
from Agents.Lester import Lester
from Utils.Preprocessing import Preprocessing
from Agents.DataObjects.TrainingData import TrainingData
from Generators.SineWave import SineWave
from numpy import ndarray
from Agents.Schools.SequentialSchool import SequentialSchool
from Agents.DataObjects.TrainingRules import TrainingRules
from Agents.ModelWrappers.SequentialSimulation import SequentialSimulator
from Visualizers.Graph import Graph


if __name__ == '__main__':

    # Synthesizing Data
    wave: SineWave = SineWave()
    wave.initialize(100)
    x: ndarray = wave.series()
    y: ndarray = wave.generate(noise=True)

    # Building Data Objects
    data: TrainingData = TrainingData(floatAllX=x, floatAllY=y, dtype=float)
    rulesTrain: TrainingRules = TrainingRules(epochs=10, plotLoss=True, verbose=1, batchSize=1000)
    rulesCompiler: CompilerRules = CompilerRules(lossFunction="mse", optimizer="adam")

    # Getting Data Ready
    data = Preprocessing.trainTestSplit(ratio=0.3, data=data)
    data = Preprocessing.reshapeLSTM(data=data, maxSampleSize=200, features=1)
    # Dynamic reshape data. Store data in one of rules set about shape of data. Use to build models, outputs.
    # Building Models
    lester: Lester = Lester()

    # TODO: Setze diese in das Datenobjekt
    lester.initialize(inputShape=(data.floatTrainX.shape[1], 1), returnSequence=True)
    lester.addSingleLSTM(20, returnSequence=False)
    lester.addSingleDense(20)
    lester.addSingleDense(20)
    lester.addDenseRange(start=19, end=2, step=5, activation="relu")
    lester.addSingleDense(1)

    # Training Models
    school: SequentialSchool = SequentialSchool(model=lester.get())
    school.initialize(rulesCompiler)
    school.fit(data, rulesTrain)
    result: ndarray = SequentialSimulator.predictSinsoid(sampleSize=200, model=school.get(), features=1)
    Graph.build(x, y, title="Initial Data")
    Graph.build([x for x, _ in enumerate(result)][200:], result[200:], title="Predicted Only")
    Graph.build([x for x, _ in enumerate(result)], result, title="Prediction Target, and Predicted")

