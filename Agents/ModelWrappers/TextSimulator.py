from numpy import zeros, asarray, log, exp, random, argmax
from Agents.DataObjects.TextData import TextData
from random import randint
from tensorflow.keras import Sequential


class TextSimulator:

    @staticmethod
    def _sample(preds, temperature=1.0):
        preds = asarray(preds).astype('float64')
        preds = log(preds) / temperature
        exp_preds = exp(preds)
        preds = exp_preds / sum(exp_preds)
        probas = random.multinomial(1, preds, 1)
        return argmax(probas)

    @staticmethod
    def generate(data: TextData, model: Sequential, diversity: float):
        start = randint(0, data.textSize - data.maxLen - 1)
        generated = ''
        sentence = data.text[start: start + data.maxLen]
        generated += sentence
        for i in range(400):
            x = zeros((1, data.maxLen, len(data.chars)))
            for t, char in enumerate(sentence):
                x[0, t, data.charIndices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = TextSimulator._sample(preds, diversity)
            next_char = data.indicesChar[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char
        print(generated)