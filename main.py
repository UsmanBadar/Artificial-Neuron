bias = 2
weight = 0.1
learning_rate = 0.1
epochs = 20
inputs = [1,2,3,4,5]
targets = [4,5,6,7,8]

def predict(i):
  return weight * i + bias


for _ in range(epochs):
  predictions = [predict(i) for i in inputs]
  errors = [(p-t)**2 for p,t in zip(predictions, targets)]
  cost = sum(errors)/len(inputs)
  print(f"Weight:{weight:.2f}, Bias:{bias:.2f}, Cost:{cost:.2f}")
  error_derivatives = [2*(p-t) for p,t in zip(predictions, targets)]
  weight_derivatives = [e * i for e,i in zip(error_derivatives, inputs)]
  bias_derivatives = [e*1 for e in error_derivatives]
  weight -= learning_rate * sum(error_derivatives)/len(error_derivatives)
  bias -= learning_rate * sum(bias_derivatives)/len(bias_derivatives)
