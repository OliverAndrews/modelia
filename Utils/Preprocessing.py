from Agents.DataObjects.TrainingData import TrainingData
from math import floor
from numpy import array, ndarray
from typing import List


class Preprocessing:

    @staticmethod
    def trainTestSplit(ratio: float, data: TrainingData) -> TrainingData:

        if data.intAllX is not None:
            if len(data.intAllX) != len(data.floatAllY):
                raise IndexError("Train and test sets are of differing size")
        if data.floatAllX is not None:
            if len(data.floatAllX) != len(data.floatAllY):
                raise IndexError("Train and test sets are of differing size")
        if data.dtype is (not float or not int):
            raise TypeError("Invalid Type")

        if data.dtype is float:
            sizeTest: int
            sizeTrain: int
            if data.floatAllX is not None:
                sizeTest = floor(len(data.floatAllX) * ratio)
                if sizeTest > len(data.floatAllX):
                    raise IndexError("Split ratio must be less than one")
                sizeTrain = len(data.floatAllX) - sizeTest
                data.floatTestY = data.floatAllY[:sizeTest]
                data.floatTestX = data.floatAllX[:sizeTest]
                data.floatTrainX = data.floatAllX[:sizeTrain]
                data.floatTrainY = data.floatAllY[:sizeTrain]
        else:
            sizeTest: int
            sizeTrain: int
            if data.intAllX is not None:
                sizeTest: int = floor(len(data.intAllX) * ratio)
                if sizeTest > len(data.intAllX):
                    raise IndexError("Split ratio must be less than one")
                sizeTrain = len(data.intAllX) - sizeTest
                data.intTestY = data.intAllY[:sizeTest]
                data.intTestX = data.intAllX[:sizeTest]
                data.intTrainX = data.intAllX[:sizeTrain]
                data.intTrainY = data.intAllY[:sizeTrain]

        return data

    @staticmethod
    def reshapeLSTM(data: TrainingData, maxSampleSize, features: int) -> TrainingData:
        if data.intAllX is not None:
            if len(data.intAllX) != len(data.floatAllY):
                raise IndexError("Train and test sets are of differing size")
        if data.floatAllX is not None:
            if len(data.floatAllX) != len(data.floatAllY):
                raise IndexError("Train and test sets are of differing size")
        if data.dtype is (not float or not int):
            raise TypeError("Invalid Type")

        if data.dtype is float:
            if data.floatTrainX is not None:
                sampleSize: int
                for i in reversed((range(maxSampleSize + 1))):
                    if len(data.floatAllX) % i == 0:
                        sampleSize = i
                        break
                yTrain: ndarray = array(data.floatTrainY)
                newX: List[List[float]] = []
                newY: List[float] = []
                for step in range(0, len(yTrain)):
                    if (step + sampleSize) >= len(yTrain):
                        break
                    subsample = yTrain[step:step + sampleSize]
                    newX.append(subsample)
                    newY.append(yTrain[step + sampleSize])
                XArr = array(newX)
                YArr = array(newY)
                data.floatTrainX = XArr.reshape((XArr.shape[0], XArr.shape[1], features))
                data.floatTrainY = YArr

        elif data.dtype is int:
            if data.intTrainX is not None:
                pass

        return data
