from numpy import sin, linspace, ndarray, array
from random import uniform

class SineWave:
    duration: int
    timeSeries: ndarray

    def initialize(self, samples: int) -> None:
        self.timeSeries = linspace(0, samples, num=samples)

    def series(self) -> ndarray:
        return self.timeSeries

    def generate(self, noise: bool = False) -> ndarray:
        if not noise:
            return array([sin(x) for x in self.timeSeries])
        return array([sin(x) + uniform(0, 1) for x in self.timeSeries])
