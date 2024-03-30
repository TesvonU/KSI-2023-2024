import csv
import joblib

#Načetení vytrénovaného modelu a dat pro predikci
model = joblib.load('model.pkl')
file_path = input("Zadej cestu k data inputu: ")

'''
přečtení dat ze souboru, kód postupně čte řádky, převádí true/false na 1/0, stringy s čísly na int
nakonec zvolí relevantní data pro predikci (zvolil jsem nsfw, spoiler a ups)
'''
test_data = []
with open(file_path, 'r', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=';')
    for line in csvreader:          #kód předpokládá, že na prvním řádku jsou data, nikoliv názvy labelů
        for column in line:
            if column == 'false':
                line[line.index(column)] = 0
            if column == 'true':
                line[line.index(column)] = 1
        nsfw, spoiler, ups = int(line[4]), int(line[5]), int(line[7])
        test_data.append([nsfw, spoiler, ups])

#vytvoření a vrácení predikce (používá se metoda uložená v modelu)
predictions = model.predict(test_data)
for prediction in predictions:
    print(prediction)
