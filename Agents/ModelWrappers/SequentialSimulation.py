from tensorflow.keras import Sequential
from Generators.SineWave import SineWave
from numpy import append
from numpy import ndarray
from Visualizers.Graph import Graph


class SequentialSimulator:
    model: Sequential

    def __init__(self, model):
        self.model = model

    @staticmethod
    def predictSinsoid(sampleSize: int, features: int, model: Sequential) -> ndarray:
        # Basierend auf dieser:
        # https://medium.com/@krzysztofbalka/training-keras-lstm-to-generate-sine-function-2e3c0ca42c3b
        wave: SineWave = SineWave()
        wave.initialize(10)
        test_xaxis = wave.series()
        calc_y = wave.generate() # Das waren falsche Daten. Modell ist eindeutig überangepasst
        # Warum macht dies Dreieck, dann richtig?
        # Ich denke, die Daten sind falsch geladen.
        test_y = calc_y[:sampleSize]
        Graph.build(list(range(1, sampleSize + 1)), test_y, title="Prediction Target")
        for i in range(len(test_xaxis) - sampleSize):
            net_input = test_y[i: i + sampleSize]
            # Das waren falsche Daten. Das Modell ist auch seltsam
            net_input = net_input.reshape((1, sampleSize, features))
            y = model.predict(net_input, verbose=1)
            test_y = append(test_y, y)
        return test_y
