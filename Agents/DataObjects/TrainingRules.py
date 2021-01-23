from dataclasses import dataclass


@dataclass
class TrainingRules:

    batchSize: int
    epochs: int
    plotLoss: bool = False
    verbose: int = 1

    learningRate: int = None

    # Multiuse variables
    intContainerOne: int = None
    intContainerTwo: int = None

    floatContainerOne: float = None
    floatContainerTwo: float = None
