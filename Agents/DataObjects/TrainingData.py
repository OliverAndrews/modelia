from dataclasses import dataclass
from typing import List


@dataclass
class TrainingData:
    dtype: type

    floatAllX: List[float] = None
    floatAllY: List[float] = None

    intAllX: List[int] = None
    intAllY: List[int] = None

    intTrainX: List[int] = None
    intTrainY: List[int] = None

    intTestX: List[int] = None
    intTestY: List[int] = None

    floatTrainX: List[float] = None
    floatTrainY: List[float] = None

    floatTestX: List[float] = None
    floatTestY: List[float] = None

