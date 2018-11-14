import csv

# open file
f = open("train_rna_combined.csv", 'rw')
csv_f = csv.reader(f)

rowNum = 0
for row in csv_f:
    row[0] = rowNum
    rowNum += 1

f.close()