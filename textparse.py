import numpy as np
import CharProcessing


class TextParse(object):

    charP = CharProcessing.CharProcessing()

    def splitText(self, text, sequenceLength):
        formatedLines = []
        currentLine = ""
        for word in text.split():
            word = word.lower()
            if not self.containsOnlySupportedCharacters(word) or len(word) > sequenceLength:
                continue
            elif len(word) + len(currentLine) <= sequenceLength:
                currentLine += word + " "
            else:
                formatedLines.append(currentLine[:len(currentLine) - 1])
                currentLine = word + " "
        if currentLine != "":
            formatedLines.append(currentLine[:len(currentLine) - 1])
        print(formatedLines)#
        return formatedLines

    def containsOnlySupportedCharacters(self, word):
        supported = True
        for char in word:
            supported = supported and self.charP.isSupported(char)
        return supported

    def lineToMatrix(self, line, sequenceLength):
        M = np.zeros((1, sequenceLength, self.charP.getSupportedNoOfLetters()))
        for i in range(len(line)):
            M[0, i] = self.charP.getOneHotForChar(line[i])
        return M

    def getInputMatrix(self, lines, sequenceLength, language):
        X = self.lineToMatrix(lines[0], sequenceLength)
        y = self.charP.getOneHotForLang(language)
        charNo = self.charP.getSupportedNoOfLetters();
        m = np.ones((1, charNo))
        for i in range(1, len(lines)):
            line = self.lineToMatrix(lines[i], sequenceLength)
            X = np.concatenate((X, line), axis = 0)
            y = np.concatenate((y, self.charP.getOneHotForLang(language)), axis = 0)
            m = np.concatenate((m, np.ones((1, charNo))), axis = 0)

        return (X, y, m)

    def getInputMatrixFromText(self, text, sequenceLength, language):
        return self.getInputMatrix(self.splitText(text, sequenceLength), sequenceLength, language)