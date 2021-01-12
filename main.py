import csv
import tensorflow as tf

from sklearn.model_selection import train_test_split


def save_data_to_file(x, y):
    with open('data/normalized.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(y)):
            writer.writerow(x[i] + [y[i]])
    csvfile.close()


def strict_normalization(x, y):
    X = list()
    Y = list()
    print("Applying strict normalization...")
    data = dict()
    for i in range(len(y)):
        if not (y[i] in data.keys()):
            data[y[i]] = list()
        data[y[i]].append(x[i])
    m = min([len(data[k]) for k in data.keys()])
    for i in range(m):
        for k in data.keys():
            X.append(data[k][i])
            Y.append(k)
    save_data_to_file(X, Y)
    return X, Y


def test(tas=5):
    X = list()
    Y = list()
    try:
        csvfile = open("data/normalized.csv", "r")
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) > 1:
                X.append([float(cell) for cell in row[:-1]])
                Y.append(int(row[-1]))
        csvfile.close()
    except IOError:
        # Read data in from file
        with open("banknotes/banknotes.csv") as f:
            reader = csv.reader(f)
            next(reader)

            data = []
            for row in reader:
                data.append({
                    "evidence": [float(cell) for cell in row[:4]],
                    "label": 1 if row[4] == "0" else 0
                })

        # Separate data into training and testing groups
        evidence = [row["evidence"] for row in data]
        labels = [row["label"] for row in data]
        X, Y = strict_normalization(evidence, labels)
    X_training, X_testing, y_training, y_testing = train_test_split(
        X, Y, test_size=0.3
    )


test()
