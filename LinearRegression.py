import numpy as np
import random

# indexes for model
weights = [0, 0, 0]


# retrieving data from the file
def get_data(file_data):
    file = open(file_data, 'r')
    file_data = file.readlines()
    file.close()
    del file_data[0]
    return file_data


# making data comfortable for using
def parse_data(str_set):
    for counter, data in enumerate(str_set):
        data = data.split()
        del data[0]
        data.insert(1, 1)
        str_set[counter] = list(map(float, data))
    return str_set


# divides data into 2 arrays : train and validate\test
def divide_data(data_set):
    rand_set = []
    for n in range(0, int(len(data_set) * 1 / 7)):
        data = random.choice(data_set)
        data_set.remove(data)
        rand_set.append(data)
    return rand_set


# predicting result - get results according to weights and features
def prediction(data, weights):
    return np.matmul(data, weights)


# cost function - counts diff between prediction and real result
def cost_func(data_set, weights):
    res = 0
    for data in data_set:
        res += (prediction(data[1:], weights) - data[0]) ** 2
    return res / (len(data_set))


def cost_derivative(data_set, weights):
    res = 0
    for data in data_set:
        res += (prediction(data[1:], weights) - data[0]) * data[1]
    return res / len(data_set)


#gradient descent
def gradient_descent(data_set, weights, step):
    while True:
        prev = cost_func(data_set, weights)
        for n, w in enumerate(weights):
            weights[n] = w - (step * cost_derivative(data_set, weights) / len(data_set))
        if prev - cost_func(data_set, weights) < 0.01:
            break
    return weights


def polynomial(data, weights):
    sum = 0;
    for n, d in enumerate(data):
        sum += pow(d, n) * weights[n]
    return sum;


def regularization(data_set, weights):
    return

if __name__ == '__main__':
    strt_data = get_data('house_prices.txt')
    strt_data = parse_data(strt_data)
    valid_set = divide_data(strt_data)
    print(cost_func(strt_data, weights))
    #print(str(prediction(data_set[0][1:], weights)) + "  " + str(data_set[0][0]))
    #print(str(prediction(data_set[10][1:], weights)) + "  " + str(data_set[10][0]))
    print(gradient_descent(strt_data, weights, 0.001))
    print(cost_func(strt_data, weights))
    for data in valid_set:
        print(str(prediction(data[1:], weights)) + "   " + str(data[0]))
