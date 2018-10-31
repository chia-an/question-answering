import pickle
import numpy as np
from keras import backend as K
from consts import model_dir, data_dir
from sklearn.metrics import recall_score


def keras_recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    
    Copied from https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def keras_precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    
    Copied from https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def keras_f1(y_true, y_pred):
    """F1 metric.
    
    Copied from https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    """
    precision = keras_precision(y_true, y_pred)
    recall = keras_recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def recall_at_k(y_pred, k=2, group_size=2):
    """Compute the recall@k metric.
    
    Modify from https://github.com/basma-b/dual_encoder_udc/blob/master/utilities/data_helper.py
    """
    y_pred = y_pred.flatten()
    n_options = 10
    n_test = len(y_pred) // n_options
    n_correct = 0
    for i in range(n_test):
        preds = np.array(y_pred[i*n_options: (i+1)*n_options])[:group_size]
        indices = np.argpartition(preds, -k)[-k:]
        if 0 in indices:
            n_correct += 1
    return float(n_correct) / n_test




def load_dictionary():
    """Load dictionary and create inverted index.
    
    The data is downloaded from https://drive.google.com/file/d/1VjVzY0MqKj0b-q_KXnaHC49qCw3iDqEY/view
    """
    MAX_SEQUENCE_LENGTH, _, word_index = pickle.load(open(data_dir / 'params.pkl', 'rb'))
    n_words = len(word_index) + 1  # 1-indexed

    inv_word_index = {}
    for word, index in word_index.items():
        assert index not in inv_word_index, index
        inv_word_index[index] = word
    
    return word_index, inv_word_index, MAX_SEQUENCE_LENGTH



def load_data(ratio=0.01):
    """Load the splitted data.
    
    The data is downloaded from https://drive.google.com/file/d/1VjVzY0MqKj0b-q_KXnaHC49qCw3iDqEY/view
    """
    print("Loading data with sampling ratio", ratio)
    train_c, train_r, train_l = pickle.load(open(data_dir / 'train.pkl', 'rb'))
    dev_c, dev_r, dev_l = pickle.load(open(data_dir / 'dev.pkl', 'rb'))
    test_c, test_r, test_l = pickle.load(open(data_dir / 'test.pkl', 'rb'))
    dev_l = np.array(dev_l)
    test_l = np.array(test_l)
    
    n_train = int(ratio * train_c.shape[0])
    n_dev   = int(ratio * dev_c.shape[0])
    n_test  = int(ratio * test_c.shape[0])
    
    train_c, train_r, train_l = train_c[:n_train, :], train_r[:n_train, :], train_l[:n_train]
    dev_c, dev_r, dev_l = dev_c[:n_dev, :], dev_r[:n_dev, :], dev_l[:n_dev]
    test_c, test_r, test_l = test_c[:n_test, :], test_r[:n_test, :], test_l[:n_test]
    print("# training data:", n_train)
    print("# validation data:", n_dev)
    print("# testing data:", n_test)
    
    
    return train_c, train_r, train_l, \
           dev_c,   dev_r,   dev_l, \
           test_c,  test_r,  test_l



def dataset_to_corpus(data, dictionary):
    return [' '.join([dictionary[token] for token in x if token != 0])
            for x in data]
