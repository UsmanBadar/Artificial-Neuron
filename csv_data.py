import csv

training_data = [
    ['Weight', 'Age', 'Blood Fat Content'],
    [84, 46, 354],
    [73, 20, 190],
    [65, 52, 405],
    [70, 30, 263],
    [76, 57, 451],
    [69, 25, 302],
    [63, 28, 288],
    [72, 36, 385],
    [79, 57, 402],
    [75, 44, 365],
    [27, 24, 209],
    [89, 31, 290],
    [65, 52, 346],
    [57, 23, 254],
    [59, 60, 395],
    [69, 48, 434],
    [60, 34, 220],
    [79, 51, 374],
    [75, 50, 308],
    [82, 34, 220]
]


test_data = [
    ['Weight', 'Age', 'Blood Fat Content'],
    [59, 46, 311],
    [67, 23, 181],
    [85, 37, 274],
    [55, 40, 303],
    [63, 30, 244]    
]


def create_datasets(file_name, data):
    with open(file_name, 'w', newline='') as csvFile:
        csvWriter = csv.writer(csvFile)
        for row in data:
            csvWriter.writerow(row)

create_datasets('training_data.csv', training_data)
create_datasets('test_data.csv', test_data)
