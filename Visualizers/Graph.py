import matplotlib.pyplot as plt


class Graph:
    data: dict

    def __init__(self, data: dict) -> None:
        self.data = data

    def build(self):
        plt.plot(self.data["x"], self.data["y"])
        plt.show()
