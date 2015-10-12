import numpy as np

LANG_NO = 3

class CharProcessing(object):

    def __init__(self):
        self.charArray =['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '\'', 'ä', 'ö', 'ü', 'ß', 'à', 'á', 'è', 'é', 'ì', 'í', 'ò', 'ó', 'ù', 'ú', 'î', 'ñ', 'ő', 'ű', 'â', 'ã', 'ê', 'ô', 'õ', 'ç'];
        self.languages = ['EN', 'DE', 'IT', 'HU', 'ES', 'NL', 'PT']
        #Lines above process 7 languages: English, German, Italian, Hungarian, Spanish, Dutch and Portugese.
        # If you want to process only three languages, you should use instead lines below (English, German and Italian).
        #self.charArray =['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '\'', 'ä', 'ö', 'ü', 'ß', 'à', 'á', 'è', 'é', 'ì', 'í', 'ò', 'ó', 'ù', 'ú', 'î'];
        #self.languages = ['EN', 'DE', 'IT']

    def isSupported(self, charFromText):
        return charFromText in self.charArray

    def getSupportedIndex(self, charFromText):
        try:
            result = self.charArray.index(charFromText)
            return result
        except:
            return -1;

    def getSupportedNoOfLetters(self):
        result = len(self.charArray)
        return result

    def getOneHotForChar(self, c):
        result = np.zeros((1, len(self.charArray)))
        index = self.charArray.index(c)
        result[0, index] = 1
        return result

    def getOneHotForLang(self, lang):
        result = np.zeros((1, LANG_NO))
        index = self.languages.index(lang)
        result[0, index] = 1
        return result