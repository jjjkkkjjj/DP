import csv
from csvReader import csvReader

def main():
    with open('pitch-type.csv', 'r') as f:
        reader = csv.reader(f)
        for type in reader:
            # read data
            csvfiles = type[2:]

            # read all data and hold it to DataList
            for csvfile in csvfiles:
                data = csvReader(csvfile, type[0])
                data.save('__video/{0}.mp4'.format(csvfile[:-4]), fps=240, saveonly=True)


if __name__ == '__main__':
    main()