import os
import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


class Model:
    def __init__(self, num_input, num_classes):
        self.num_input = num_input
        self.num_classes = num_classes
        self.X = tf.placeholder("float", [None, self.num_input])
        self.Y = tf.placeholder("float", [None, self.num_classes])
        self.model = self.create_model()
        print('Model has been created')

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

    def train(self, version, data, labels, batch_size, test_data, test_labels):
        beta = 0.5
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.Y) +
                              beta + tf.nn.l2_loss(self.weights['h2']) + beta + tf.nn.l2_loss(self.biases['b2']) +
                              beta + tf.nn.l2_loss(self.weights['out']) + beta + tf.nn.l2_loss(self.biases['out']))
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
                    print('Epoch: ' + str(epoch) + ' Accuracy: ' + '{:.3f}'.format(avg_acc/num_batches)
                      + ' Loss: ' + '{:.4f}'.format(avg_loss/num_batches))

            print('Optimization Finished!')
            print('Testing Accuracy:',
                  sess.run(accuracy, feed_dict={self.X: test_data, self.Y: test_labels}))
            if not os.path.isdir(str(version)):
                os.mkdir(str(version))
            save_path = saver.save(sess, str(version) + '/model.ckpt')
            print("Model saved in file: %s" % save_path)

    def test(self, version,test_data, test_labels):
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


class MNIST_Data:
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

    def reduce_data(self, percentage):
        self.train_x, self.predict_x, self.train_y, self.predict_y = \
            train_test_split(self.train_x, self.train_y, test_size=percentage)

    def increase_data(self, inputs, num_to_label):
        training_data_x = self.train_x
        training_data_y = self.train_y

        maxes = np.zeros(len(inputs))
        for i in range(len(inputs)):
            maxes[i] = inputs[i][np.argmax(inputs[i])]
        for i in range(num_to_label):
            index = (min(enumerate(maxes)))[0]
            maxes[index] = 0.
            predict = self.predict_x[index]
            training_data_x = np.append(training_data_x, predict)
            training_data_y = np.append(training_data_y, [self.predict_y[index]])
            self.predict_x = np.delete(self.predict_x, index)
            self.predict_y = np.delete(self.predict_y, index)
        self.train_x = training_data_x
        self.train_y = training_data_y


def main():
    model = Model(784, 10)
    data = MNIST_Data('mnist_train.csv', 'mnist_test.csv')
    original_size = len(data.train_x)
    print('\nOriginal Size: ' + str(original_size))
    model.train(0, data.train_x, data.train_y, 100, data.test_x, data.test_y)
    data.reduce_data(0.99)
    for i in range(1, 11):
        print('\nVersion ' + str(i) + ' Size: ' + str(len(data.train_x)))
        model.train(i, data.train_x, data.train_y, 100, data.test_x, data.test_y)
        predictions = model.predict(i, data.predict_x)
        data.increase_data(predictions, int(original_size * 0.01))
    model.test(i, data.test_x, data.test_y)


if __name__ == '__main__':
    main()
