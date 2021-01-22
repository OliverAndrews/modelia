from dataclasses import dataclass
from numpy import ndarray
from typing import List, Dict


@dataclass
class TextData:

    maxLen: int = None
    steps: int = None
    text: List[str] = None
    textSize: int = None

    testX: ndarray = None
    testY: ndarray = None

    chars: List[str] = None
    charIndices: Dict[str, int] = None
    indicesChar: Dict[int, str] = None
    nextChars: List[str] = None

    sentences: List[str] = None
