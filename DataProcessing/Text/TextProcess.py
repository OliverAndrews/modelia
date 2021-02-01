from Agents.DataObjects.TextData import TextData
from Agents.DataObjects.TextDataWordLevel import TextDataWordLevel
from numpy import zeros
from requests import get
from bs4 import BeautifulSoup
from typing import List
import string


class TextProcess:

    def __init__(self):
        pass

    @staticmethod
    def vectorizeTextWeb(url: str, data: TextData, chunkSize: int = 3) -> TextData:
        """
                        Taken and modified from
        ===========================================================

        https://gist.githubusercontent.com
        /mostafa-mahmoud/b7058bb8e5b2079ad1cb45d0873de67d
        /raw/3d075fbab2a1dc60b6bacbf4f96930f2f0b7ae73/nietzsche.py
        """
        text: str = " ".join(list(map(lambda x: x.get_text().replace("\r", "").replace("\n", "").lower(),
            list(BeautifulSoup(get(url).text, "html.parser").findAll("p")))))
        data.text = text
        data.textSize = len(text)
        data.chars = sorted(list(set(text)))
        data.charIndices = dict((c, i) for i, c in enumerate(data.chars))
        data.indicesChar = dict((i, c) for i, c in enumerate(data.chars))

        data.maxLen = 40
        data.step = 3
        data.sentences = []
        data.nextChars = []
        for i in range(0, len(text) - data.maxLen, chunkSize):
            data.sentences.append(text[i: i + data.maxLen])
            data.nextChars.append(text[i + data.maxLen])

        data.testX = zeros((len(data.sentences), data.maxLen, len(data.chars)), dtype=bool)
        data.testY = zeros((len(data.sentences), len(data.chars)), dtype=bool)

        for i, sentence in enumerate(data.sentences):
            for t, char in enumerate(sentence):
                data.testX[i, t, data.charIndices[char]] = 1
            data.testY[i, data.charIndices[data.nextChars[i]]] = 1

        return data

    @staticmethod
    def vectorizeTextFile(location: str, data: TextData, chunkSize: int = 3) -> TextData:
        """
                        Taken and modified from
        ===========================================================

        https://gist.githubusercontent.com
        /mostafa-mahmoud/b7058bb8e5b2079ad1cb45d0873de67d
        /raw/3d075fbab2a1dc60b6bacbf4f96930f2f0b7ae73/nietzsche.py
        """
        text: str = open(location, "r").read()
        data.text = text
        data.textSize = len(text)
        data.chars = sorted(list(set(text)))
        data.charIndices = dict((c, i) for i, c in enumerate(data.chars))
        data.indicesChar = dict((i, c) for i, c in enumerate(data.chars))

        data.maxLen = 40
        data.step = 3
        data.sentences = []
        data.nextChars = []
        for i in range(0, len(text) - data.maxLen, chunkSize):
            data.sentences.append(text[i: i + data.maxLen])
            data.nextChars.append(text[i + data.maxLen])

        data.testX = zeros((len(data.sentences), data.maxLen, len(data.chars)), dtype=bool)
        data.testY = zeros((len(data.sentences), len(data.chars)), dtype=bool)

        for i, sentence in enumerate(data.sentences):
            for t, char in enumerate(sentence):
                data.testX[i, t, data.charIndices[char]] = 1
            data.testY[i, data.charIndices[data.nextChars[i]]] = 1

        return data

    @staticmethod
    def vectorizeTextFileWord(location: str, data: TextData, chunkSize: int = 3) -> TextData:
        text: str = open(location, "r").read()
        junk: List[str] = list(filter(lambda x: x != ".", list(string.punctuation)))
        junk.append("\n")  # Add newline character

        # Clean text of characters
        for _, char in enumerate(junk):
            text = text.replace(char, "")
        text = text.replace(".", " ")
        # Raw text
        data.text = text.split(" ")
        data.textSize = len(data.text)
        
        # Split the text at the word level
        data.chars = set(list(filter(lambda x: x != ' ', [x.lower() for x in sorted(list(set(data.text)))])))
        # Build forward and backward dicts
        data.charIndices = dict((c.lower(), i) for i, c in enumerate(data.chars))
        data.indicesChar = dict((i, c.lower()) for i, c in enumerate(data.chars))

        data.maxLen = 40
        data.step = chunkSize
        data.sentences = []
        data.nextChars = []

        # This is something I'll want to alter... looks bound
        # to the letter based structure
        for i in range(0, len(text) - data.maxLen, chunkSize):
            data.sentences.append(text[i: i + data.maxLen])
            data.nextChars.append(text[i + data.maxLen])
        data.testX = zeros((len(data.sentences), data.maxLen, len(data.chars)), dtype=bool)
        data.testY = zeros((len(data.sentences), len(data.chars)), dtype=bool)

        for i, sentence in enumerate(data.sentences):
            for t, char in enumerate(sentence.split(" ")):
                data.testX[i, t, data.charIndices[char.lower()]] = 1
            data.testY[i, data.charIndices[data.nextChars[i]]] = 1

        return data
