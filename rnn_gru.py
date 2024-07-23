import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from biosiglive import load, OfflineProcessing
from post_process_data import ProcessData
from sklearn.preprocessing import MinMaxScaler
import os
# Generate dummy time series data
# def generate_time_series(num_series, num_timesteps):
#     np.random.seed(42)
#     time = np.arange(num_timesteps)
#     series = np.array([np.sin(time + np.random.uniform(0, 2 * np.pi)) + np.random.normal(0, 0.1, size=num_timesteps) for _ in range(num_series)])
#     return series

def generate_time_series(num_series, num_timesteps):
    part = "P14"
    dlc_data_path = f"/mnt/shared/Projet_hand_bike_markerless/RGBD/P14/gear_20_22-01-2024_16_29_55{os.sep}marker_pos_multi_proc_3_crops_normal_times_three_filtered_ribs_and_cluster_pp.bio"
    # dlc_data, _, names, frame_idx = utils.load_data_from_dlc(None, dlc_data_path, part)
    data = load(dlc_data_path)
    frame_idx = data["frame_idx"]
    dlc_data = data["markers_in_meters"]
    names = list(data["markers_names"][:, 0])
    # markers_in_pixel = data["markers_in_pixel"]
    n_final = dlc_data.shape[2]
    dlc_data = dlc_data[..., :n_final]
    frame_idx = frame_idx[:n_final]
    new_markers_dlc = ProcessData()._fill_and_interpolate(data=dlc_data,
                                                          idx=frame_idx,
                                                          shape=frame_idx[-1] - frame_idx[0],
                                                          fill=True)
    new_markers_dlc_filtered = np.zeros((3, new_markers_dlc.shape[1], new_markers_dlc.shape[2]))
    for i in range(3):
        new_markers_dlc_filtered[i, :8, :] = OfflineProcessing().butter_lowpass_filter(
            new_markers_dlc[i, :8, :],
            3, 60, 2)
        new_markers_dlc_filtered[i, 8:, :] = OfflineProcessing().butter_lowpass_filter(
            new_markers_dlc[i, 8:, :],
            10, 60, 2)
    real_num_series = dlc_data.shape[2] // num_timesteps
    final_mat = np.zeros((real_num_series, num_timesteps, 3, dlc_data.shape[1]))
    target_mat = np.zeros((real_num_series, num_timesteps, 3, dlc_data.shape[1]))
    for i in range(real_num_series):
        for m in range(dlc_data.shape[1]):
            final_mat[i, :, :, m] = dlc_data[:, m, i*num_timesteps:(i+1)*num_timesteps].T
            target_mat[i, :, :, m] = new_markers_dlc_filtered[:, m, i*num_timesteps:(i+1)*num_timesteps].T
    return final_mat, target_mat, real_num_series

# Parameters
num_series = 1000
num_timesteps = 60
num_features = 1  # Each time step has only one feature
input_timesteps = 60  # Number of timesteps to use as input

# Generate data
inputs, target, num_series = generate_time_series(num_series, num_timesteps)

# Prepare input-output pairs
def create_dataset(data, target, input_timesteps):
    X, y = [], []
    for s in range(data.shape[0]):
        X.append(data[s, :-1,  0, 10])
        #y.append(series[:-1])
        y.append(target[s, -1, 0, 10])
        # plt.plot(data[s, :-1, 1, 1])
        # plt.plot(target[s, :, 1, 1])
        # plt.scatter(data.shape[1]-1, target[s, -1, 0, 1])
        # plt.scatter(data.shape[1], target[s, -1, 1, 0])
        # plt.scatter(data.shape[1], target[s, -1, 2, 0])
        plt.show()
    X = np.array(X)
    y = np.array(y)
    return X, y

X, y = create_dataset(inputs, target, input_timesteps)
X = X[...]
y = y[:, np.newaxis]
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X = scaler_x.fit_transform(X)
y = scaler_y.fit_transform(y)
X = X[..., np.newaxis]  # Add feature dimension

# Split the data into training and testing sets manually
split_index = int(len(X) * 0.8)
train_X, test_X = X[:split_index], X[split_index:]
train_y, test_y = y[:split_index], y[split_index:]

# Ensure data is in numpy array format
train_X = np.array(train_X)
test_X = np.array(test_X)
train_y = np.array(train_y)
test_y = np.array(test_y)
train = True
if train :
    # Define the GRU model
    # model = Sequential([
    #     Input(shape=(input_timesteps, num_features)),  # Define the input shape
    #     GRU(num_timesteps, activation='relu'),
    #     Dense(num_features)
    # ])
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_timesteps-1, 1)))
    model.add(layers.GRU(50, return_sequences=True, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.GRU(50, return_sequences=False))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(num_features))
    model.compile(optimizer='adam', loss='mse')


    # Train the model
    model.fit(train_X, train_y, epochs=200, batch_size=16, validation_split=0.6)
    # save model
    model.save("model.h5")
else:
    model = tf.keras.models.load_model("model.h5")
# Predict in a loop over all points of a time series
def predict_full_sequence(model, initial_sequence):
    current_sequence = initial_sequence
    predictions = []
    n_predictions = initial_sequence.shape[0]
    for n in range(n_predictions):
        prediction = model.predict(current_sequence[n:n+1, :, ...], verbose=0)
        predictions.append(prediction[0, 0])

        # Update the current sequence to include the new prediction
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1, 0] = prediction

    return np.array(predictions)
#load model
# Select one test series to predict
# Predict the remaining points in the series
predicted_series = predict_full_sequence(model, test_X)
test_X = scaler_x.inverse_transform(test_X[..., 0])
predicted_series = scaler_y.inverse_transform(predicted_series[:, np.newaxis])
test_y = scaler_y.inverse_transform(test_y)
# Plot the results
plt.figure(figsize=(12, 6))
for i in range(test_X.shape[0]):
    plt.plot(test_X[i, :], label='Actual')
    plt.scatter(test_X.shape[1], test_y[i, 0], c="r")
    plt.scatter(test_X.shape[1], predicted_series[i], label='prediction', alpha=0.5, c="g")
    #plt.scatter(input_timesteps, test_y[i], label='Actual', c='r')
#plt.legend()
plt.title('Time Series Prediction')
plt.show()
