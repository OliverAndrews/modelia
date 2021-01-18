from dataclasses import dataclass
from typing import List


@dataclass
class CompilerRules:
    lossFunction: str
    optimizer: str
