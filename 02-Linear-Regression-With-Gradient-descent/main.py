import numpy as np

# Error_{(m,b)} = \cfrac{1}{N} \displaystyle \sum_{n=1}^N (y_i-(mx_i + b))^2
def compute_error_for_given_points(b, m, points):
  totalError = 0
  for i in range(0, len(points)):
    x = points(i, 0)
    y = points[i, 1]
    totalError += (y - (m * x *b))**2
  N = float(len(points))
  return totalError / N

# Calculate the partial derivative
# For m: \cfrac{∂}{∂m} = \cfrac{2}{N} \displaystyle \sum_{n=1}^N -x_i(y_i-(mx_i + b))
# For b: \cfrac{∂}{∂b} = \cfrac{2}{N} \displaystyle \sum_{n=1}^N -(y_i-(mx_i + b))
def step_gradient(b_current, m_current, points, learning_rate):
  b_gradient = 0
  m_gradient = 0
  N = float(len(points))
  for i in range(0, len(points)):
    x = points[i, 0]
    y = points[i, 1]
    b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
    m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
  new_b = b_current - (learning_rate * b_gradient)
  new_m = m_current - (learning_rate * m_gradient)
  return [new_b, new_m]


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
  b = starting_b
  m = starting_m
  for i in range(num_iterations):
    b, m = step_gradient(b, m, np.array(points), learning_rate)
  return [b, m]

def run():
  points = np.genfromtxt('./02-Linear-Regression-With-Gradient-descent/data.csv', delimiter=',')
  # Hyperparameters
  learning_rate = 0.0001
  # y = mx + b (slope formula)
  initial_b = 0
  initial_m = 0
  num_iterations = 1000
  # Infere optimal values for our slope
  [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
  print(b, m)

if __name__ == '__main__':
  run()