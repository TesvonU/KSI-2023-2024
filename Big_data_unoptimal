import pandas as pd
import csv


#UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8a in position 14: invalid start byte


INPUT_FILE = "measurements.txt"



def main():
    batch_size = 5000000
    output = pd.DataFrame(columns=['Stanice', 'min', 'max', 'avg'])
    for batch in pd.read_csv(INPUT_FILE, names=['Stanice', 'Teplota'], sep=';', encoding='utf-8', chunksize=batch_size):
        batch_stat = batch.groupby('Stanice')['Teplota'].agg(['min', 'max', 'mean']).reset_index()
        output = pd.concat([output, batch_stat], ignore_index=True)
    output = output.groupby('Stanice').agg({'min': 'min', 'max': 'max', 'mean': 'mean'}).sort_values(by='Stanice').reset_index()
    output['mean'] = round(output['mean'], 1)
    output.to_csv('output.csv', sep=';', index=False, header=False, encoding="utf-8")

    with open('output.csv', 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file, delimiter=';')
        for row in csv_reader:
            print(';'.join(row))

            
if __name__ == '__main__':
    main()
