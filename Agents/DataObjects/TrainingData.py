from dataclasses import dataclass
from typing import List


@dataclass
class TrainingData:
    dtype: type

    intTrainX: List[int]
    intTrainY: List[int]

    intTestX: List[int]
    intTestY: List[int]

    floatTrainX: List[float]
    floatTrainY: List[float]

    floatTestX: List[float]
    floatTestY: List[float]

