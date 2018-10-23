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
            name = Dir.split('/')[-2]
            # read all data and hold it to DataList
            for csvfile in csvfiles:
                data = csvReader(csvfile, type[0])
                DataLists.append(data)

            # detect reference
            referenceDetector(DataLists, name + '-' + type[1] + '.csv', superDir=name)


if __name__ == '__main__':
    main()