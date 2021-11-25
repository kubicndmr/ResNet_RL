import csv

counter = 0

with open('labels_train.csv', newline='') as t:

    spamreader = csv.reader(t, delimiter='\n')

    for i, row in enumerate(spamreader):
        if i % 10 == 0:
            if i > 0:   
                row = row[0].split(',')
                row[0] = str(counter)
                counter += 1
            print(row)

            with open('labels_valid.csv', 'a', newline='') as v:
                writer = csv.writer(v)
                writer.writerow(row)
    t.close()
'''

with open('labels_valid.csv', newline='') as t:
    spamreader = csv.reader(t, delimiter='\n')

    for i, row in enumerate(spamreader):
        pass

    print(i)

'''
