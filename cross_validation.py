import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gc

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
# import seaborn as sns

from models.Conv1d import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold


def value_of_mat(mat_filename):
    """
    load the mat file and return the data.
    sio.loadmat returns a dict and 'val' means value.
    """

    return sio.loadmat(mat_filename)["val"][0, :]


def len_of_mat(mat_filename):
    return len(value_of_mat(mat_filename))


def plot_ecg(mat_filename, time_interval=1000):
    ecg_signal = list(value_of_mat(mat_filename))
    plt.plot(ecg_signal[:time_interval])


def num2onehot(number, length):
    x = np.zeros(length)
    x[number] = 1
    return x


def num2onehot_for_list(a_list):
    length = max(a_list) + 1
    return np.array([num2onehot(number, length) for number in a_list])


def onehot2num_for_list(onehot_array):
    return [list(onehot).index(1) for onehot in onehot_array]


def duplicate_padding(signals, UB_LEN_MAT):
    """
    padding the signals not with zeros but the copy of the signal.

    :param: signals: list of np.array with 1 dimension.
        more general, it should be a list of objects, which has length and can be concatenate.
    :param: UB_LEN_MAT: int
    """

    X = np.zeros((len(signals), UB_LEN_MAT))
    for i, sig in enumerate(signals):
        if len(sig) >= UB_LEN_MAT:
            X[i, :] = sig[0: UB_LEN_MAT]
        else:
            sig_copy_section = sig[0: (UB_LEN_MAT - len(sig))]
            sig_replay = np.hstack((sig, sig_copy_section))  # np.concatenate()

            # concatenate copied signal to original signal until its length meets the upper bound
            while len(sig_replay) < UB_LEN_MAT:
                sig_copy_section = sig[0:(UB_LEN_MAT - len(sig_replay))]
                sig_replay = np.hstack((sig_replay, sig_copy_section))

            X[i, :] = sig_replay
    return X


def train_and_evaluate(model, X_train, Y_train, X_test, Y_test, model_name):

    model_path = "./trained_models/{}/".format(model_name)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    checkpointer = ModelCheckpoint(filepath='{}/best_model.h5'.format(model_path),
                                   monitor='val_acc',
                                   verbose=1,
                                   save_best_only=True)

    # early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=50, verbose=1, mode='auto')

    # print("x shape", X_train.shape)
    # print("y shape", Y_train.shape)

    hist = model.fit(X_train, Y_train,
                     validation_data=(X_test, Y_test),
                     batch_size=275,
                     epochs=3,
                     verbose=2,
                     shuffle=True,
                     callbacks=[checkpointer])

    # save history
    pd.DataFrame(hist.history).to_csv('{}/history.csv'.format(model_path))

    pd.DataFrame(hist.history['acc']).to_csv('{}/train_acc.csv'.format(model_path))
    pd.DataFrame(hist.history['loss']).to_csv('{}/loss.csv'.format(model_path))
    pd.DataFrame(hist.history['val_acc']).to_csv('{}/val_acc.csv'.format(model_path))
    pd.DataFrame(hist.history['val_loss']).to_csv('{}/val_loss.csv'.format(model_path))

    # evaluation
    predictions = model.predict(X_test)

    score = accuracy_score(onehot2num_for_list(Y_test), predictions.argmax(axis=1))
    print('Last epoch\'s validation score is ', score)

    df = pd.DataFrame(predictions.argmax(axis=1))
    df.to_csv('{}/preds_{.4f}.csv'.format(model_path, score), index=None, header=None)

    cm = confusion_matrix(onehot2num_for_list(Y_test), predictions.argmax(axis=1))
    df = pd.DataFrame(cm)
    df.to_csv('{}/confusion_matrix_{.4f}.csv'.format(model_path, score), index=None, header=None)

    del model
    K.clear_session()
    gc.collect()

    return score


def cross_validation(model, X, Y, n_fold=10):
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True)
    y = Y.reshape(X.shape[0],)

    scores = []
    for i, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train = X[train_index, :]
        Y_train = Y[train_index, :]
        X_val = X[val_index, :]
        Y_val = Y[val_index, :]
        score = train_and_evaluate(model, X_train, Y_train, X_val, Y_val, i)
        scores.append(score)

    return scores


# project parameters
DATA_PATH = 'data/training2017/'
LABELS_PATH = DATA_PATH + 'REFERENCE.csv'

# lower bound of the length of the signal
LB_LEN_MAT = 100

# upper bound of the length of the signal
UB_LEN_MAT = 10100

LABELS = ["N", "A", "O"]
n_classes = len(LABELS) + 1

np.random.seed(7)

if __name__ == "__main__":
    
    # this helps a lot when debugging
    print(os.getcwd())

    # step 1: get data
    files = [f for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f))]
    mat_files = [f for f in files if f.startswith("A") and f.endswith('.mat')]

    # filter out short mat_files
    mat_files = [f for f in mat_files if len_of_mat(os.path.join(DATA_PATH, f)) >= LB_LEN_MAT]

    signals = [value_of_mat(os.path.join(DATA_PATH, f)) for f in mat_files]
    signal_IDs = [f.split(".")[0] for f in mat_files]

    n_sample = len(signal_IDs)
    print('Total training size is ', n_sample)

    # get X
    X = duplicate_padding(signals, UB_LEN_MAT)

    # get Y
    df_label = pd.read_csv(LABELS_PATH, sep=',', header=None, names=None)
    df_label.columns = ["sigID", "label"]
    df_label = df_label.set_index("sigID")

    labels = [df_label.loc[sigID, "label"] for sigID in signal_IDs]
    label_ids = [LABELS.index(l) if l in LABELS else 3 for l in labels]

    Y = num2onehot_for_list(label_ids)

    # data preprocessing
    X = (X - X.mean()) / (X.std())
    X = np.expand_dims(X, axis=2)

    # shuffle the data
    values = [i for i in range(len(X))]
    permutations = np.random.permutation(values)
    X = X[permutations, :]
    Y = Y[permutations, :]

    # train test split
    train_test_ratio = 0.9

    X_train = X[:int(train_test_ratio * n_sample), :]
    Y_train = Y[:int(train_test_ratio * n_sample), :]
    X_test = X[int(train_test_ratio * n_sample):, :]
    Y_test = Y[int(train_test_ratio * n_sample):, :]
    
    # load the model and train it
    model = conv1d(UB_LEN_MAT)

    cross_validation(model, X, Y)
    # train_and_evaluate(model, X_train, Y_train, X_test, Y_test, "conv_model")

