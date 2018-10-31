import time
import pickle

from keras import backend as K
from keras import Sequential, Model
from keras.layers import Embedding, LSTM, Input, Dense, Dot, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from utils import keras_recall, keras_precision, keras_f1, load_data, load_dictionary
from consts import model_dir, model_log_dir

from numpy.random import seed
seed(0)

from tensorflow import set_random_seed
set_random_seed(0)


def build_model():
    encode_dim = 50

    # Load word embedding
    with open(model_dir / 'wid_embedding.pkl', 'rb') as f:
        wid_embedding = pickle.load(f)
    n_words, embedding_dim = wid_embedding.shape  # (482089, 300)
    
    _, _, MAX_SEQUENCE_LENGTH = load_dictionary()
    
    
    context_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    response_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    
    encoder = Sequential()
    encoder.add(Embedding(input_dim=n_words,
                          input_length=MAX_SEQUENCE_LENGTH,
                          output_dim=embedding_dim,
                          weights=[wid_embedding],
                          mask_zero=True,
                          trainable=True))
    encoder.add(LSTM(units=encode_dim))

    # Dual encoder / siamese-like architecture
    context_branch = encoder(context_input)
    response_branch = encoder(response_input)

    # sigmoid(c * (x1 * M * x2^T) + b)
    context_branch = Dense(units=encode_dim, use_bias=False)(context_branch)
    dot_product = Dot(axes=1)([context_branch, response_branch])
    out = Dense(units=1, activation='sigmoid')(dot_product)

    # sigmoid(x1 * x2^T)
    # dot_product = Dot(axes=1)([context_branch, response_branch])
    # out = Activation('sigmoid')(dot_product)
    
    
    dual_encoder = Model([context_input, response_input], out)
    dual_encoder.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[
            keras_recall,
            keras_precision,
            keras_f1,
            'accuracy'
        ],
    )

    return dual_encoder


def train():
    n_epochs = 100
    batch_size = 128

    train_c, train_r, train_l, \
    dev_c, dev_r, dev_l, \
    test_c, test_r, test_l = load_data(ratio=0.05)
    
    
    dual_encoder = build_model()
    dual_encoder.fit(
        x=[train_c[:, :], train_r[:, :]],
        y=train_l[:],
        batch_size=batch_size,
        epochs=n_epochs,
        callbacks=[
            ModelCheckpoint(
                filepath=str(model_dir / 'model.{epoch}.{val_loss}.hdf5'),
                period=1,
            ),
            ModelCheckpoint(
                filepath=str(model_dir / 'best.hdf5'),
                save_best_only=True,
            ),
            TensorBoard(
                log_dir=str(model_log_dir),
                update_freq='batch',
            ),
        ],
        validation_data=([dev_c[:, :], dev_r[:, :]],
                         dev_l[:]),
        verbose=1,
    )
    
    filename = 'dual_encoder_{}.h5'.format(time.strftime('%y%m%d_%H%M%S'))
    print('Save the model to', filename)
    dual_encoder.save(str(model_dir / filename))


if __name__ == '__main__':
    train()
