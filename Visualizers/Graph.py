import matplotlib.pyplot as plt
plt.style.use('dark_background')

class Graph:

    @staticmethod
    def build(x, y, title: str):
        plt.plot(x, y)
        plt.title(title)
        plt.show()
