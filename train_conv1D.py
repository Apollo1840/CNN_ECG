import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, accuracy_score

from models.Conv1d import *
from utils_data import *
from utils_ml import *

np.random.seed(7)

# project parameters
DATA_PATH = 'data/training2017/'

# lower bound of the length of the signal
LB_LEN_MAT = 100

# upper bound of the length of the signal
UB_LEN_MAT = 10100

LABELS = ["N", "A", "O"]
n_classes = len(LABELS) + 1

if __name__ == "__main__":
    
    # this helps a lot when debugging
    print(os.getcwd())

    X, Y = load_cinc_data(DATA_PATH, LB_LEN_MAT, LABELS)

    # data preprocessing
    X = duplicate_padding(X, UB_LEN_MAT)
    X = (X - X.mean()) / (X.std())
    X = np.expand_dims(X, axis=2)

    # train test split
    train_test_ratio = 0.9
    n_sample = X.shape[0]

    X_train = X[:int(train_test_ratio * n_sample), :]
    Y_train = Y[:int(train_test_ratio * n_sample), :]
    X_test = X[int(train_test_ratio * n_sample):, :]
    Y_test = Y[int(train_test_ratio * n_sample):, :]
    
    # load the model and train it
    model = conv1d(UB_LEN_MAT)

    checkpointer = ModelCheckpoint(filepath='./trained_models/Best_model.h5',
                                   monitor='val_acc',
                                   verbose=1,
                                   save_best_only=True)

    # print("x shape", X_train.shape)
    # print("y shape", Y_train.shape)

    hist = model.fit(X_train, Y_train,
                     validation_data=(X_test, Y_test),
                     batch_size=275,
                     epochs=3,
                     verbose=2,
                     shuffle=True,
                     callbacks=[checkpointer])
    
    # evaluation
    predictions = model.predict(X_test)

    score = accuracy_score(onehots2numbers(Y_test), predictions.argmax(axis=1))
    print('Last epoch\'s validation score is ', score)

    df = pd.DataFrame(predictions.argmax(axis=1))
    df.to_csv('./trained_models/Preds_' + str(format(score, '.4f')) + '.csv', index=None, header=None)

    confusion_matrix = confusion_matrix(onehots2numbers(Y_test), predictions.argmax(axis=1))
    df = pd.DataFrame(confusion_matrix)
    df.to_csv('./trained_models/Result_Conf' + str(format(score, '.4f')) + '.csv', index=None, header=None)
