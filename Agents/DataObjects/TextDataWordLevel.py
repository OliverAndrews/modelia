from dataclasses import dataclass
from numpy import ndarray
from typing import List, Dict


@dataclass
class TextDataWordLevel:

    maxLen: int = None
    steps: int = None
    text: List[str] = None
    textSize: int = None

    testX: ndarray = None
    testY: ndarray = None

    chars: List[str] = None # Words, change later
    charIndices: Dict[str, int] = None # Forward dict
    indicesChar: Dict[int, str] = None # Reverse dict
    nextChars: List[str] = None

    sentences: List[str] = None
