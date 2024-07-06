import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# read from files
train_data = pd.read_csv('TRAIN.csv')
test_data = pd.read_csv('TEST.csv')

# split data from target attribute ANGLE-ACC-ARM
#  y represents dependent variable and X represents independent variables
y_train = train_data['ANGLE-ACC-ARM']
y_test = test_data['ANGLE-ACC-ARM']

X_train = train_data.drop('ANGLE-ACC-ARM', axis=1)
X_test = test_data.drop('ANGLE-ACC-ARM', axis=1)

# no need to normalize because every data in data set between 0 and 1

# set number of hidden layers and their sizes
model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# calculate prediction to find mae, mse, rmse and R^2 for test and train
predictionsTrain = model.predict(X_train)
maeTrain = mean_absolute_error(y_train, predictionsTrain)
mseTrain = mean_squared_error(y_train, predictionsTrain)
rmseTrain = np.sqrt(mseTrain)
r2Train = r2_score(y_train, predictionsTrain)

predictionsTest = model.predict(X_test)
maeTest = mean_absolute_error(y_test, predictionsTest)
mseTest = mean_squared_error(y_test, predictionsTest)
rmseTest = np.sqrt(mseTest)
r2Test = r2_score(y_test, predictionsTest)

# write performance scores in report
with open('report.txt', 'w') as file:
    file.write("Train results:\n")
    file.write(f"MAE: {maeTrain}\n")
    file.write(f"MSE: {mseTrain}\n")
    file.write(f"RMSE: {rmseTrain}\n")
    file.write(f"R² (Coefficient of Determination): {r2Train}\n")
    file.write("Test results:\n")
    file.write(f"MAE: {maeTest}\n")
    file.write(f"MSE: {mseTest}\n")
    file.write(f"RMSE: {rmseTest}\n")
    file.write(f"R² (Coefficient of Determination): {r2Test}\n\n")

    file.write(f"Parameters :\n")
    file.write(f"**Number of hidden layer = {len((model.coefs_))-1}\n")
    file.write(f"**Activation name = {model.activation}\n")
    file.write(f"**Initial learning rate = {model.learning_rate_init}\n")
    file.write(f"**Momentum = {model.momentum}\n")
    file.write(f"**Gradient method = {model.solver}\n")

    file.write(f"\nWEIGHTS:\n")
    # weights between layers
    for i, layer_weights in enumerate(model.coefs_):
        print(i)
        shape = layer_weights.shape
        if i == 0:
            file.write(f"\nWeights between input and layer{i+1}\n")
        elif i == 1:
            file.write(f"\nWeights between layer{i} and layer{i + 1}\n")
        else:
            file.write(f"\nWeights between layer{i} and output\n")
        for j in range(shape[0]):
            if i == 0:
                file.write(f" from x{j} to each node in hidden layer{i + 1} weights:\n")
            elif i == 1:
                file.write(f" from node{j} in hidden layer{i} to each node in hidden layer{i+1} weights:\n")
            else:
                file.write(f" from node{j} in hidden layer{i} to output node weights:")

            file.write(np.array2string(layer_weights[j], precision=4, separator=', ') + '\n')
    file.write(f"\nBIASES:\n")
    # biases in nodes
    for i, layer_biases in enumerate(model.intercepts_):
        if i == 2:
            file.write(f"\nBiases in output node \n")
        else:
            file.write(f"\nBiases in layer{i+1} nodes \n")
        for j in range(layer_biases.shape[0]):
            if i == 2:
                file.write(f"output bias: ")
            else:
                file.write(f"node{j} bias in layer{i+1}: ")
            file.write(np.array2string(layer_biases[j], precision=4, separator=', ') + '\n')

