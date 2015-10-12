import DataSet


def createDataSets(type):
    #Code below creates datasets for seven languages, but you should comment last for in case you want to use only three languages
    dataSet = DataSet.DataSet()
    textEn = open(type + "En.txt", "r")
    dataSet.add(textEn.readline(), 'EN')
    textEn.close()
    textDe = open(type + "De.txt", "r")
    dataSet.add(textDe.readline(), 'DE')
    textDe.close()
    textIt = open(type + "It.txt", "r")
    dataSet.add(textIt.readline(), 'IT')
    textIt.close()
    textHu = open(type + "Hu.txt", "r")
    dataSet.add(textHu.readline(), 'HU')
    textHu.close()
    textNl = open(type + "Nl.txt", "r")
    dataSet.add(textNl.readline(), 'NL')
    textNl.close()
    textEs = open(type + "Es.txt", "r")
    dataSet.add(textEs.readline(), 'ES')
    textEs.close()
    textPt = open(type + "Pt.txt", "r")
    dataSet.add(textPt.readline(), 'PT')
    textPt.close()
    dataSet.store(type[0] + "Set_7_15.ds") #here you should put name of your file
    dataSet.shuffle()
    dataSet.store(type[0] + "Set_s_7_15.ds") #here you should put name of your file; data stored in those files are shuffled
    #better use these sets for training and validation to make sure that your results are not biased

    return dataSet

def main():
    testingSet = createDataSets("training")
    validationSet = createDataSets("validation")

if __name__ == '__main__':
    main()
