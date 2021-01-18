from Agents.Lester import Lester
from Utils.Preprocessing import Preprocessing
from Agents.DataObjects.TrainingData import TrainingData
from Generators.SineWave import SineWave
from numpy import ndarray


if __name__ == '__main__':

    # Synthesizing Data
    wave: SineWave = SineWave(5000)
    wave.initialize(5000)
    x: ndarray = wave.series()
    y: ndarray = wave.generate()

    # Building Data Objects
    data: TrainingData = TrainingData(
        floatAllX=x,
        floatAllY=y,
        dtype=float)

    # Getting Data Ready
    data = Preprocessing.trainTestSplit(0.3, data)
    data = Preprocessing.reshapeLSTM(data=data, maxSampleSize=200, features=1)

    """
    # Building Models
    lester: Lester = Lester()
    lester.initialize()  # Look into specifying the input shape vs not input shape.
    lester.addSingleLSTM(5)
    lester.addSingleDense(3)
    model = lester.get()
    model.compile(optimizer="adam", loss="mse")
    model.fit(data.floatTrainX, data.floatTrainY, epochs=20, verbose=1)

    # Training Models
    """

