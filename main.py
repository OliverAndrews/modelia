from Agents.Lester import Lester
from Utils.Preprocessing import Preprocessing
from Agents.DataObjects.TrainingData import TrainingData
from Generators.SineWave import SineWave
from typing import List


if __name__ == '__main__':

    # Synthesizing Data
    wave: SineWave = SineWave(10000)
    wave.initialize(10000)
    x: List[float] = list(wave.series())
    y: List[float] = list(wave.generate())

    # Building Data Objects
    data: TrainingData = TrainingData(
        floatTrainX=x,
        floatTrainY=y,
        dtype=float)


    # Getting Data Ready
    data = Preprocessing.trainTestSplit(0.3, data)


    # Building Models
    lester: Lester = Lester()
    lester.addSingleLSTM(128)
    lester.addSingleDense(10)

    # Building Schools

    # Training Models

