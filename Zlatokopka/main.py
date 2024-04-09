import cv2 as cv
import torch
from torchvision import transforms
import pkg_resources

from dataset import FashionDataset
from model import FashionModel, HParams, activations

##########################################################################################
##                                                                                      ##
##                                    UPOZORNĚNÍ                                        ##
##                                                                                      ##
##   Příklady jsou zvoleny tak, aby demonstrovaly, jak pracovat s poskytnutými          ##
##   třídami. Rozhodně není optimální, co se týče volby parametrů, jejich hledání,      ##
##   způsobu učení, načítání dat a podobně. Pro tyto účely byste si měli napsat         ##
##   vlastní kód a kód v příkladech (bez úprav) nepoužívat při hledání optimálních      ##
##   hyperparametů                                                                      ##
##                                                                                      ##  
##########################################################################################



##########################################################################
##     PŘÍKLAD 1: UČENÍ MODELU S DEFAULTNÍMI PARAMETRY NA 1 OBRÁZKU     ##
##########################################################################

def example1():
    hParams = HParams()

    trainSet = FashionDataset(cacheToRAM=True, maxImagesPerClass=1, classes=["Bags"]) # Vytvořím dataset obsahující pouze 1 obrázek ze třídy Bags
    model = FashionModel(hParams, trainSet.labels) # Vytvořím novou instanci nenatrénovaného modelu

    model.train(trainSet, epochs=10) # provedu 10 iterací učení modelu

    loss, accuracy = model.evaluate(trainSet) # Ohodnotím, jak je model naučený na stejném obrázku, jako na kterém byl trénovaný (accuracy by měla být až 100 % (1.0) a loss 0.0)
    print(loss, accuracy)


#####################################################################################################################
##     PŘÍKLAD 2: UČENÍ MODELU S BATCH-SIZE=8, JINAK S DEFAULTNÍMI PARAMETRY NA 10 OBRÁZCÍCH ZE 3 RŮZNÝCH TŘÍD     ##
#####################################################################################################################

def example2():
    hParams = HParams(batchSize=8)

    trainSet = FashionDataset(cacheToRAM=True, maxImagesPerClass=10, classes=["Bags", "Ties", "Dress"])
    validationSet = trainSet.extractSubset(0.1, balanceClasses=False) # Vytvořím validační dataset z 10 % trénovacího datasetu (bude obsahovat 3 obrázky)

    model = FashionModel(hParams, trainSet.labels)

    for i in range(10): # provedu 100 iterací učení modelu, kde každých 10 iterací ohodnotím model na validačním datasetu a uložím jej
        model.train(trainSet, epochs=10, showEpochProgressBar=False, showTotalProgressBar=True) # progres bar nebude pro každou epochu, ale pro celý trénovací proces 10 epoch
        valLoss, valAccuracy = model.evaluate(validationSet) # Ohodnotíme model na validačním datasetu
        savedPath = model.save("model_epoch_{}".format(10 + i * 10)) # uložíme model
        print("Model saved at: " + savedPath + "\nValidation Loss: {}, Validation Accuracy: {}".format(valLoss, valAccuracy))


#####################################################################################################################
##                                  PŘÍKLAD 3: URČENÍ NEJLEPŠÍ AKTIVAČNÍ FUNKCE                                    ##
#####################################################################################################################
def example3():
    hParams = HParams(learningRate=0.01)
    trainSet = FashionDataset(cacheToRAM=False, size=(720, 540)) # Načtu celý dataset a změním velikost obrázků
    validationSet = trainSet.extractSubset(0.1)

    bestActivation, bestAccuracy = None, 0.0

    for activation in [activations.LEAKY_RELU, activations.RELU, activations.SIGMOID, activations.TANH]: # Vyzkouším všechny aktivace
        print("Testing: " + str(activation)) 

        hParams.activation = activation # Upravím hyperparametry
        model = FashionModel(hParams, trainSet.labels, imageSize=(720, 540)) # Vytvořím novou instanci modelu s novými hyperparametry (nutné model informovat o změně defaultních rozměrů obrázku!!)

        model.train(trainSet, epochs=10) # Natrénuji model v 10 iteracích s aktuálními hyperparametry
        valLoss, valAccuracy = model.evaluate(validationSet) # Ohodnotím dataset
        savedPath = model.save(str(activation))
        print("Model saved at: " + savedPath + "\nValidation Loss: {}, Validation Accuracy: {}".format(valLoss, valAccuracy))
        if valAccuracy > bestAccuracy:
            bestActivation, bestAccuracy = activation, valAccuracy

    print("Best activation: " + str(bestActivation) + " with accuracy: " + str(bestAccuracy))

#####################################################################################################################
##                                        PŘÍKLAD 4: PRÁCE S DATASETEM                                             ##
#####################################################################################################################
def example4():
    dataset = FashionDataset(cacheToRAM=False, size=(720, 540), transform=transforms.RandomVerticalFlip(1)) # Všechny obrázky v datasetu budou vertikálně převráceny
    print("Třídy v datasetu: ", dataset.labels)
    print("Počet dat v datasetu: ", len(dataset)) 
    fifthImage = FashionDataset.tensorToOpenCVImage(dataset[4]["img"]) # Pátý obrázek z datasetu
    cv.imshow("Fifth image", fifthImage)
    cv.waitKey(0)
    print("Třída 5. obrázku: ", dataset.labels[torch.argmax(dataset[4]["label"]).item()])


#####################################################################################################################
##                                     PŘÍKLAD 5: POUŽITÍ NAUČENÉHO MODELU                                         ##
#####################################################################################################################
def example5():
    model = FashionModel.load("activations.LEAKY_RELU") # Pro načtení jiného modelu upravte název souboru, předpokládám, že jste spustili example3() a model tak již existuje
    print(model.predict(cv.imread("dataset/Bags/0001.jpg"))) # Predikce libovolného obrázku
    print(model.predictBatch([cv.imread("dataset/Bags/0001.jpg"), cv.imread("dataset/Dress/0001.jpg")])) # Predikce více obrázků
    dataset = FashionDataset(cacheToRAM=True, maxImagesPerClass=2, classes=model.classes, size=model.imageSize) # NUTNÉ, ABY BYLA VELIKOST OBRÁZKŮ A POLÉ TŘÍD STEJNÉ JAKO V DATASETU, NA KTERÉM BYL MODEL TRÉNOVÁN    
    print(model.predict(dataset[0]["img"])) # Predikce obrázku z datasetu
    print(model.predictBatch([dataset[0]["img"], dataset[1]["img"]])) # Predikce více obrázků z datasetu
    print(model.evaluate(dataset)) # Ohodnocení modelu na datasetu


##########################################################################
##                OTESTUJE VERZE NAINSTALOVANÝCH KNIHOVEN               ##
##########################################################################


def testLibsVersions():
    required_packages = [
        ('tqdm', 'any'),
        ('torch', '2.1.1'),
        ('torchvision', '0.16.1'),
        ('numpy', '1.25.1')
    ]
    
    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])

    for package in required_packages:
        if package[1] == 'any':
            if not any(p.startswith(package[0]) for p in installed_packages_list):
                print(f"Package '{package[0]}' is NOT installed.")
        else:
            if not f"{package[0]}=={package[1]}" in installed_packages_list:
                print(f"Package '{package[0]}' is NOT installed with required version '{package[1]}'.")


##########################################################################
##              ODKOMENTUJTE, KTERÝ PŘÍKLAD CHCETE SPUSTIT              ##
##########################################################################

# example1()
# example2()
# example3()
# example4()
# example5()
testLibsVersions()
