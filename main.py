from Agents.Lester import Lester


if __name__ == '__main__':
    lester: Lester = Lester()
    lester.addSingleLSTM(128)
    lester.addSingleDense(10)
    print(lester)
