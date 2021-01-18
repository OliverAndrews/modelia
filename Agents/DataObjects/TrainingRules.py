from dataclasses import dataclass


@dataclass
class TrainingRules:

    batchSize: int
    epochs: int
    plotLoss: bool = False
    verbose: int = 1
