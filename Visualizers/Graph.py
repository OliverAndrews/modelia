import matplotlib.pyplot as plt


class Graph:

    @staticmethod
    def build(x, y):
        plt.plot(x, y)
        plt.show()
