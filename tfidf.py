import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

from utils import load_data, load_dictionary, dataset_to_corpus, recall_at_k
from consts import data_dir


def tfidf():
    print('Loading data ...')
    train_c, train_r, train_l, \
    dev_c, dev_r, dev_l, \
    test_c, test_r, test_l = load_data(ratio=0.05)
    
    n_dev = dev_c.shape[0]
    n_test = test_c.shape[0]

    
    # Transform to text corpus for tfidf vectorizer
    print('Transforming to text corpuses ...')
    _, inv_word_index, _ = load_dictionary()
    
    train_c_corpus = dataset_to_corpus(train_c, inv_word_index)
    train_r_corpus = dataset_to_corpus(train_r, inv_word_index)
    dev_c_corpus   = dataset_to_corpus(dev_c, inv_word_index)
    dev_r_corpus   = dataset_to_corpus(dev_r, inv_word_index)
    test_c_corpus  = dataset_to_corpus(test_c, inv_word_index)
    test_r_corpus  = dataset_to_corpus(test_r, inv_word_index)
    
    
    # Transform to tfidf vector
    print('Fitting a tfidf model ...')
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(train_c_corpus + train_r_corpus)
    
    print('Transforming to tfidf features ...')
    X_dev_c = vectorizer.transform(dev_c_corpus)
    X_dev_r = vectorizer.transform(dev_r_corpus)
    X_test_c = vectorizer.transform(test_c_corpus)
    X_test_r = vectorizer.transform(test_r_corpus)

    
    # Tfidf predict (cosine similarity)
    print('Predicting ...')
    
    def cos_sim(x, y):
        return x * y.transpose() / np.sqrt(x * x.transpose() * y * y.transpose())
    
    y_pred_dev = np.array([cos_sim(X_dev_c[i, :], X_dev_r[i, :])
                           for i in range(n_dev)])
    y_pred_test = np.array([cos_sim(X_test_c[i, :], X_test_r[i, :])
                            for i in range(n_test)])
    
    
    # Evaluate results
    print('Results:\n')
    #print('Validation set')
    #for group_size in [2, 10]:
    #    for k in [1, 2, 5]:
    #        if k >= group_size:
    #            break
    #        r = recall_at_k(y_pred_dev, k, group_size)
    #        print('recall@{} ({} options): {}'.format(k, group_size-1, r))
    #print('Testing set')
    for group_size in [2, 10]:
        for k in [1, 2, 5]:
            if k >= group_size:
                break
            r = recall_at_k(y_pred_test, k, group_size)
            print('recall@{} ({} options): {}'.format(k, group_size-1, r))
    

if __name__ == '__main__':
    tfidf()
    