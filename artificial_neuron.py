import csv

training_filename = 'training_data.csv'
test_filename = 'test_data.csv'

training_data = {}
test_data = {}
bias = 25
weights = [0.7, 0.9]
learning_rate = 0.012
epochs = 700


def read_data_file(filename, data):
    with open(filename, mode = 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for col_name in header:
            data[col_name] = []
        
        for row in csv_reader:
            for i, col_name in enumerate(header):
                value = int(row[i])
                data[col_name].append(value)


read_data_file(training_filename, training_data)
read_data_file(test_filename, test_data)


def normalize_data(data, *args):
    for i in args:
        min_value = min(data[i])
        max_value = max(data[i])
        normalized_row = [round((value - min_value)/(max_value - min_value), 3)for value in data[i]]
        data[i] = normalized_row


normalize_data(training_data, 'Weight', 'Age')
normalize_data(test_data, 'Weight', 'Age')


def create_datasets(data, target, *args):
    inputs = list(zip(*[data[col] for col in args]))
    targets = data[target]
    return inputs, targets


training_inputs, training_targets = create_datasets(training_data, 'Blood Fat Content', 'Weight', 'Age')
test_inputs, test_targets = create_datasets(test_data, 'Blood Fat Content', 'Weight', 'Age')


def predict(inp, weights, bias):
  return sum([weight * i for weight, i in zip(weights, inp)]) + bias


def gradient_descent(inputs, targets, w, b, l, e):
    for _ in range(e):
        predictions = [predict(i, w, b) for i in inputs]
        error_derivatives = [2*(p-t) for p,t in zip(predictions, targets)]
        weight_derivatives = [[e * i for i in inp] for e,inp in zip(error_derivatives, inputs)]
        weight_derivatives_T = list(zip(*weight_derivatives))
        bias_derivatives = [e*1 for e in error_derivatives]
        for i in range(len(w)):
            w[i] -= l * sum(weight_derivatives_T[i])/len(weight_derivatives)
        b -= l * sum(bias_derivatives)/len(bias_derivatives)
    return w, b


final_weights, final_bias = gradient_descent(training_inputs, training_targets, weights, bias, learning_rate, epochs)


def print_result():
    result = [round(predict(i, final_weights, final_bias), 3) for i in test_inputs]
    print('\n')
    for i, target in zip(result, test_targets):
        print(f"Target: {target}, Prediction: {i}\n")

print_result()



