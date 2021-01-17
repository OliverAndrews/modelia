from Utils.Mute import Mute


def silent(func):
    def wrapper():
        Mute.muteTensorflow()
        return func()

    return wrapper
