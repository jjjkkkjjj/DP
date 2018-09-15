import csv
from dp.dp import referenceDetector
from csvReader import csvReader

def main():
    with open('pitch-type.csv', 'r') as f:
        reader = csv.reader(f)
        for type in reader:
            # read data
            DataLists = []
            csvfiles = type[2:]
            Dir = type[0]
            # read all data and hold it to DataList
            for csvfile in csvfiles:
                data = csvReader(csvfile, type[0])
                DataLists.append(data)

            # detect reference
            referenceDetector(DataLists, type[0].split('/')[-2] + type[1] + '.csv')


if __name__ == '__main__':
    main()