import matplotlib.pyplot as plt
plt.style.use('dark_background')

class Graph:

    @staticmethod
    def build(x, y):
        plt.plot(x, y)
        plt.show()
