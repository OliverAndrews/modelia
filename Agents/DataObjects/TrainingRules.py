from dataclasses import dataclass


@dataclass
class TrainingRules:

    batchSize: int
    epochs: int
