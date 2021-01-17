from numpy import sin, linspace, ndarray


class SineWave:
    duration: int
    timeSeries: ndarray

    def __init__(self, duration) -> None:
        self.duration = duration

    def initialize(self, samples: int) -> None:
        self.timeSeries = linspace(0, self.duration, num=samples)

    def series(self) -> ndarray:
        return self.timeSeries

    def generate(self) -> ndarray:
        return sin(self.timeSeries)
