from dataclasses import dataclass
from numpy import ndarray


@dataclass
class LSTMData:
    dtype: type

    intAll: ndarray = None
    intTrain: ndarray = None
    intTest: ndarray = None
    intTrainY: ndarray = None
    intTestY: ndarray = None

    floatAll: ndarray = None
    floatTrain: ndarray = None
    floatTest: ndarray = None
    floatTrainY: ndarray = None
    floatTestY: ndarray = None
