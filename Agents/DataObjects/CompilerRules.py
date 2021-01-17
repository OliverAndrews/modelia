from dataclasses import dataclass
from typing import List


@dataclass
class CompilerRules:
    lossFunction: object
    optimizerString: str
    metrics: List[str]
