import csv
import time

import tensorflow as tf
import numpy as np

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


@tf.function
def train_step(x, y, model, optimizer, loss_fn, train_acc_metric):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value


@tf.function
def test_step(x, y, model, val_acc_metric):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)


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
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3
    )

    inputs = tf.keras.Input(shape=(4,))
    x1 = tf.keras.layers.Dense(8, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(10, name="predictions")(x1)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    # Instantiate a loss function.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Prepare the training dataset.
    batch_size = 5
    x_train = np.reshape(x_train, (-1, 4))
    x_test = np.reshape(x_test, (-1, 4))
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(64)
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    epochs = 20
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step(x_batch_train, y_batch_train, model, optimizer, loss_fn, train_acc_metric)

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %d samples" % ((step + 1) * 64))

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            test_step(x_batch_val, y_batch_val, model, val_acc_metric)

        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))

    # model.compile(
    #     optimizer="adam",
    #     loss="binary_crossentropy",
    #     metrics=["accuracy"]
    # )
    #
    # model.save('models/banknotes.h5')
    #
    # model.evaluate(np.array(x_test), np.array(y_test), verbose=1)


test()
