import os
import csv
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as k
from itertools import product
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# 0 = No data balancing, 1 = Balanced data selction, 2 = Weighted cost function
balance = 2
# 0 = No Fine Tuning, 1 = Data selection adds to training data, 2 = Data selection becomes training set
fineTuning = 0


class Model:
    def __init__(self, num_input, num_classes):
        self.num_input = num_input
        self.num_classes = num_classes
        self.X = tf.placeholder("float", [None, self.num_input])
        self.Y = tf.placeholder("float", [None, self.num_classes])
        self.weights, self.biases = {}, {}
        self.model = self.create_model()

    def create_model(self):
        self.weights = {
            'h1': tf.Variable(tf.truncated_normal([self.num_input, 256], stddev=1e-2)),
            'h2': tf.Variable(tf.truncated_normal([256, 256], stddev=1e-2)),
            'out': tf.Variable(tf.truncated_normal([256, self.num_classes], stddev=1e-2))
        }
        self.biases = {
            'b1': tf.Variable(tf.truncated_normal([256], stddev=1e-2)),
            'b2': tf.Variable(tf.truncated_normal([256], stddev=1e-2)),
            'out': tf.Variable(tf.truncated_normal([self.num_classes], stddev=1e-2))
        }

        layer_1 = tf.add(tf.matmul(self.X, self.weights['h1']), self.biases['b1'])
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        out_layer = tf.matmul(layer_2, self.weights['out']) + self.biases['out']
        return out_layer
    
    @staticmethod
    def weighted_crossentropy(y_true, y_pred, weights):
        nb_cl = len(weights)
        final_mask = k.zeros_like(y_pred[..., 0])
        y_pred_max = k.max(y_pred, axis=-1)
        y_pred_max = k.expand_dims(y_pred_max, axis=-1)
        y_pred_max_mat = k.equal(y_pred, y_pred_max)
        for c_p, c_t in product(range(nb_cl), range(nb_cl)):
            w = k.cast(weights[c_t, c_p], k.floatx())
            y_p = k.cast(y_pred_max_mat[..., c_p], k.floatx())
            y_t = k.cast(y_pred_max_mat[..., c_t], k.floatx())
            final_mask += w * y_p * y_t
        return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred) * final_mask

    def train(self, version, data, labels, batch_size, test_data, test_labels, weights=np.ones((0, 0))):
        beta = 0.5
        if balance == 2:
            loss = self.weighted_crossentropy(y_true=self.Y, y_pred=self.model, weights=weights)
        else:
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.Y)
        loss += beta + tf.nn.l2_loss(self.weights['h2']) + beta + tf.nn.l2_loss(self.biases['b2'])
        loss += beta + tf.nn.l2_loss(self.weights['out']) + beta + tf.nn.l2_loss(self.biases['out'])
        loss = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
        correct_pred = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            data = np.split(data, batch_size)
            labels = np.split(labels, batch_size)
            num_batches = len(data)
            for epoch in range(1, 100 + 1):
                avg_loss, avg_acc = 0, 0
                for batch in range(num_batches):
                    _, cost, acc = sess.run([optimizer, loss, accuracy],
                                            feed_dict={self.X: data[batch], self.Y: labels[batch]})
                    avg_loss += cost
                    avg_acc += acc
                if epoch % 10 == 0:
                    message = 'Epoch: ' + str(epoch) + ' Accuracy: ' + '{:.3f}'.format(avg_acc/num_batches)
                    message += ' Loss: ' + '{:.4f}'.format(avg_loss/num_batches)
                    print(message)
            final_acc = sess.run(accuracy, feed_dict={self.X: test_data, self.Y: test_labels})
            predictions = sess.run(tf.nn.softmax(self.model), feed_dict={self.X: test_data})
            self.confusion_matrix(predictions, test_labels)
            print('Optimization Finished!')
            print('Testing Accuracy: ', str(final_acc))
            if not os.path.isdir(str(version)):
                os.mkdir(str(version))
            save_path = saver.save(sess, str(version) + '/model.ckpt')
            print("Model saved in file: %s" % save_path)
            return final_acc

    def test(self, version, test_data, test_labels):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, str(version) + '/model.ckpt')
            correct_pred = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            print('Testing Accuracy:',
                  sess.run(accuracy, feed_dict={self.X: test_data, self.Y: test_labels}))

    def predict(self, version, data):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, str(version) + '/model.ckpt')
            predictions = sess.run(self.model, feed_dict={self.X: data})
            return predictions

    @staticmethod
    def confusion_matrix(predictions, labels):
        y_actu = np.zeros(len(labels))
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if labels[i][j] == 1.00:
                    y_actu[i] = j
        y_pred = np.zeros(len(predictions))
        for i in range(len(predictions)):
            y_pred[i] = np.argmax(predictions[i])

        p_labels = pd.Series(y_pred)
        t_labels = pd.Series(y_actu)
        df_confusion = pd.crosstab(t_labels, p_labels, rownames=['Actual'], colnames=['Predicted'], margins=True)
        print('\nAccuracy = ' + str(accuracy_score(y_true=y_actu, y_pred=y_pred, normalize=True)) + '\n')
        print(df_confusion)
        print('\n' + str(classification_report(y_actu, y_pred, digits=4)))


class MNISTData:
    def __init__(self, train_file, test_file):
        train_x, train_y = [], []
        test_x, test_y = [], []
        with open(train_file) as file:
            reader = csv.reader(file)
            for row in reader:
                label = np.zeros(10)
                label[int(row[0])] = 1
                train_y.append(label)
                train_x.append(list(map(int, row[1:])))
        print('Training Data Loaded from file')

        with open(test_file) as file:
            reader = csv.reader(file)
            for row in reader:
                label = np.zeros(10)
                label[int(row[0])] = 1
                test_y.append(label)
                test_x.append(list(map(int, row[1:])))
        print('Testing Data Loaded from file')

        self.train_x, self.train_y = np.asarray(train_x), np.asarray(train_y)
        self.test_x, self.test_y = np.asarray(test_x), np.asarray(test_y)
        self.predict_x, self.predict_y = [], []

    def reduce_data(self, percentage):
        if balance == 1:
            elements = math.ceil((len(self.train_y) * percentage / 100) / 10)
            indexes = np.array([])
            for classification in range(10):
                temp_indexs = []
                for i in range(len(self.train_y)):
                    if np.argmax(self.train_y[i]) == classification:
                        temp_indexs.append(i)
                indexes = np.append(indexes, random.sample(temp_indexs, elements))
            indexes = -np.sort(-indexes)
            delete_indexes = []
            for i in range(len(self.train_x)-1, -1, -1):
                if i not in indexes:
                    self.predict_x.append(self.train_x[i])
                    self.predict_y.append(self.train_y[i])
                    delete_indexes.append(i)
            self.train_x = np.delete(self.train_x, delete_indexes, axis=0)
            self.train_y = np.delete(self.train_y, delete_indexes, axis=0)
            self.predict_x, self.predict_y = np.asarray(self.predict_x), np.asarray(self.predict_y)
        else:
            self.train_x, self.predict_x, self.train_y, self.predict_y = train_test_split(self.train_x,
                                                                                          self.train_y,
                                                                                          test_size=percentage)

    def increase_data(self, inputs, num_to_label):
        maxes = np.zeros(len(inputs))
        for i in range(len(inputs)):
            maxes[i] = inputs[i][np.argmax(inputs[i])]
        indexes = []
        if fineTuning == 2:
            self.train_x = np.zeros((0,784))
            self.train_y = np.zeros((0,10))
        if balance == 1:
            num_to_label = int(num_to_label / 10)
            classification = [[], [], [], [], [], [], [], [], [], []]
            for i in range(len(maxes)):
                prediction_class = int(np.argmax(self.predict_y[i]))
                classification[prediction_class].append([maxes[i], i])
            for class_maxes in classification:
                c_maxes, big_indexes = [m[0] for m in class_maxes], [n[1] for n in class_maxes]
                for i in range(num_to_label):
                    index = np.where(c_maxes == np.asarray(c_maxes).min())[0][0]
                    class_maxes[index][0] = np.finfo(np.float64).max
                    index = big_indexes[index]
                    self.train_x = np.vstack((self.train_x, [self.predict_x[index]]))
                    self.train_y = np.vstack((self.train_y, [self.predict_y[index]]))
                    indexes.append(index)
            self.predict_x = np.delete(self.predict_x, indexes, axis=0)
            self.predict_y = np.delete(self.predict_y, indexes, axis=0)
        else:
            for i in range(num_to_label):
                index = np.where(maxes == maxes.min())[0][0]
                maxes[index] = np.finfo(np.float64).max
                self.train_x = np.vstack((self.train_x, [self.predict_x[index]]))
                self.train_y = np.vstack((self.train_y, [self.predict_y[index]]))
                indexes.append(index)
            self.predict_x = np.delete(self.predict_x, indexes, axis=0)
            self.predict_y = np.delete(self.predict_y, indexes, axis=0)

    def get_weights(self, smooth_factor=0):
        temp_y = []
        for i in self.train_y:
            temp_y.append(np.argmax(i))
        counter = Counter(temp_y)
        if smooth_factor > 0:
            p = max(counter.values()) * smooth_factor
            for key in counter.keys():
                counter[key] += p
        majority = max(counter.values())
        weights = {cls: float(majority / count) for cls, count in counter.items()}

        nb_cl = len(weights)
        final_weights = np.ones((nb_cl, nb_cl))
        for class_idx, class_weight in weights.items():
            final_weights[0][class_idx] = class_weight
            final_weights[class_idx][0] = class_weight
        return final_weights

    def check_balance(self):
        temp_y = []
        for i in self.train_y:
            temp_y.append(np.argmax(i))
        counter = Counter(temp_y)
        print('Balance: ' + str(counter))


def main():
    data = MNISTData('mnist_train.csv', 'mnist_test.csv')
    original_size = len(data.train_x)
    print('\nOriginal Size: ' + str(original_size))
    accuracies = []
    model = Model(784, 10)
    if balance == 2:
        accuracies.append(model.train(0, data.train_x, data.train_y, 100, data.test_x, data.test_y, data.get_weights()))
    else:
        accuracies.append(model.train(0, data.train_x, data.train_y, 100, data.test_x, data.test_y))
    data.reduce_data(0.99)
    model = Model(784, 10)
    for i in range(1, 11):
        print('\nVersion ' + str(i) + ' Size: ' + str(len(data.train_x)))
        data.check_balance()
        if balance == 2:
            accuracies.append(model.train(i, data.train_x, data.train_y, 100, data.test_x, data.test_y,
                                          data.get_weights()))
        else:
            accuracies.append(model.train(i, data.train_x, data.train_y, 100, data.test_x, data.test_y))
        if i != 10:
            predictions = model.predict(i, data.predict_x)
            data.increase_data(predictions, int(original_size * 0.01))
        if fineTuning == 0:
            model = Model(784, 10)
    print('\n' + str(accuracies))


if __name__ == '__main__':
    main()
