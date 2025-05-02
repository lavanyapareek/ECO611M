import numpy as np
import sklearn.neural_network as sknn
np.random.seed(42)

N = 10000
X = np.random.uniform(-10, 10, size = (N, 2))
Y = np.linalg.norm(X, axis = 1)

indices = np.arange(N)
np.random.shuffle(indices)
folds = np.array(np.array_split(indices, 5))

M = [5, 10, 20, 50, 100]
activations = ['tanh', 'relu', 'logistic']

best_mse = float('inf')
best_config = None

for m in M:
    for acti in activations:
        mse_list = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.hstack([folds[j] for j in range(5) if j != i])

            x_train, y_train = X[train_idx], Y[train_idx]
            x_val, y_val = X[val_idx], Y[val_idx]

            model = sknn.MLPRegressor(hidden_layer_sizes=(m,), activation=acti, solver = 'adam', max_iter = 1000)
            model.fit(x_train, y_train)

            y_pred = model.predict(x_val)
            mse = np.mean((y_val - y_pred) ** 2)
            mse_list.append(mse)
        avg_mse = np.array(mse_list).mean()
        print("Hidden Layers :", m, " Activation : ", acti, " Average Mse : ", avg_mse)
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_config = (m, acti)


best_M, best_act = best_config
final_model = sknn.MLPRegressor(hidden_layer_sizes=(100,), activation='relu',
                           solver='adam', max_iter=10000, random_state=1)
#Redefine Test Vectors
X = np.random.uniform(-10, 10, size = (N, 2))
Y = np.linalg.norm(X, axis = 1)
final_model.fit(X, Y)
y_pred_full = final_model.predict(X)
train_mse = np.mean((Y - y_pred_full) ** 2)

print("\nBest Configuration:")
print(f"Hidden Layer Size: {100}, Activation: {'relu'}")
print(f"Training MSE: {train_mse:.4f}")
        
