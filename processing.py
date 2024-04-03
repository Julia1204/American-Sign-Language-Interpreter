import numpy as np
import os
import mediapipe as mp
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def preprocess_data(data_path, actions, no_sequences, sequence_length):
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(data_path, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    # print(X.shape)
    y = to_categorical(labels).astype(int)
    # print(y)
    return X, y


def split_data(X, y, test_size=0.05):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)  # podział na dane testowe - 5%
    # print(x_train.shape)
    # print(x_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)
    return x_train, x_test, y_train, y_test


log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)


def build_model(input_shape, output_classes):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))  # input_shape=(30, 1662)
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_classes, activation='softmax'))  # actions.shape[0]
    return model


# res = [0.7, 0.2, 0.1]
# actions[np.argmax(res)]

def save_model(model, model_name):
    model.save(model_name)


def train_model(model, x_train, y_train, epochs=2000, callbacks=[tb_callback]):
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(x_train, y_train, epochs, callbacks=callbacks)
    save_model(model, 'actionsv1.h5')
    model.summary()
    return model


def evaluate_model(model, x_test, y_test, actions):
    res = model.predict(x_test)
    predicted_action = actions[np.argmax(res[1])]
    print("Predicted Action:", predicted_action)
    actual_action = actions[np.argmax(y_test[1])]
    print("Actual Action:", actual_action)


# model.save('actiontest.h5')
# model.summary()

if __name__ == "__main__":
    DATA_PATH = os.path.join('Dataall')
    actions = np.array(['excuseme', 'hello', 'iloveyou', 'please', 'thanks', 'whatsup'])
    no_sequences = 80
    sequence_length = 30

    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    # res = [.7, 0.2, 0.1]
    # actions[np.argmax(res)]
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(x_train, y_train, epochs=300, callbacks=[tb_callback])
    model.save('actionsallv2.h5')
    model.summary()
