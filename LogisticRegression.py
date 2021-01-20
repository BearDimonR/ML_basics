import random

import numpy as np


# retrieving data from the file
def get_data(file_data):
    file = open(file_data, 'r')
    file_data = file.readlines()
    file.close()
    del file_data[0]
    return file_data


# parsing data
def parse_data(str, res=[]):
    for counter, data in enumerate(str):
        data = data.split(',')
        data.insert(0, 1)
        res.append((list(map(float, data))))
        if res[-1][-1] > 7:
            res[-1][-1] = 1
        else:
            res[-1][-1] = 0
    return res


# divides data into 2 arrays : train and validate\test
def divide_data(data):
    rand_set = []
    for n in range(0, int(len(data_set) * 1 / 7)):
        data = random.choice(data_set)
        data_set.remove(data)
        rand_set.append(data)
    return rand_set


# sigmoid func
def sig(n):
    return 1 / (1 + np.exp(-n))


# prediction func
def prediction(data, weights):
    return sig(np.matmul(data, weights))


def loss_func(data, weights):
    pre = prediction(data[:-1], weights)
    return (data[-1] * np.log(pre)) + (1 - data[-1]) * np.log(1 - pre)


def cost_func(data, weights):
    res = 0
    for d in data:
        res += loss_func(d, weights)
    return (- 1 / len(data)) * res


def gradient_descent(data, weights, step):
    prev = 0
    while True:
        copy = weights.copy()
        for n, w in enumerate(weights):
            for d in data:
                weights[n] = w - step / len(data) * (prediction(d[:-1], weights) - d[-1]) * d[n]
        if prev > 10:
            break
        prev += 1
    weights = copy
    return cost_func(data, weights)


if __name__ == '__main__':
    data_set = parse_data(get_data('house_data.csv'))
    test_set = divide_data(data_set)
    tettas = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    print(gradient_descent(data_set, tettas, 0.001))
    for t in test_set:
        print(str(prediction(t[:-1], tettas)) + '  ' + str(t[-1]))