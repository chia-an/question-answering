import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np
from consts import data_dir, model_dir


# Load dictionary and create inverted index
MAX_SEQUENCE_LENGTH, _, word_index = pickle.load(open(data_dir / 'params.pkl', 'rb'))
n_words = len(word_index) + 1  # 1-indexed


# Initialize
embedding_dim = 300
np.random.seed(0)
wid_embedding = np.random.rand(n_words, embedding_dim)
print('The embeding matrix has shape', wid_embedding.shape)


print('Loading the file ...')
with open(model_dir / 'glove.840B.300d.txt', 'r', encoding='utf8') as f:
    # size of the file:
    #   $ cat glove.840B.300d.txt | wc
    #   2196017 660999598 5646236541
    for line in tqdm(f):
        # The format of the line is WORD SPACE_SEPARATED_VECTOR
        word, *vector = line.rsplit(' ', maxsplit=embedding_dim)
        
        if word not in word_index:
            continue

        try:
            vector = np.array(vector, dtype=np.float32)
            assert len(vector) == embedding_dim, len(vector)
        except Exception as e:
            print(e)
            print(len(values))
            print(values[:10])
            print(line)
        
        wid_embedding[word_index[word]] = vector


print('Saving to file ...')
with open(model_dir / 'wid_embedding.pkl', 'wb') as f:
    pickle.dump(wid_embedding, f)

print('Done')
