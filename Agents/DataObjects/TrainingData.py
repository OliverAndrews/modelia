from dataclasses import dataclass
from numpy import ndarray


@dataclass
class TrainingData:
    dtype: type

    floatAllX: ndarray = None
    floatAllY: ndarray = None

    intAllX: ndarray = None
    intAllY: ndarray = None

    intTrainX: ndarray = None
    intTrainY: ndarray = None

    intTestX: ndarray = None
    intTestY: ndarray = None

    floatTrainX: ndarray = None
    floatTrainY: ndarray = None

    floatTestX: ndarray = None
    floatTestY: ndarray = None

