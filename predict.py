import keras as K
import pickle
from consts import model_dir
from utils import keras_recall, keras_precision, keras_f1, load_data


# Load the model

# model_filename = 'model.2.0.6053460450503968.hdf5'
# model_filename = 'model.3.0.6401738415222714.hdf5'
# model_filename = 'model.4.0.8720060172012735.hdf5'
model_filename = 'dot_0.05/model.2.0.5783862575187761.hdf5'
# model_filename = 'mat_0.05_dim_50/model.2.0.6308096857158685.hdf5'
# model_filename = 'mat_0.05_dim_50/model.3.0.6541781450341816.hdf5'
model = K.models.load_model(str(model_dir / model_filename),
                            custom_objects={
                                'keras_recall': keras_recall,
                                'keras_precision': keras_precision,
                                'keras_f1': keras_f1,
                            })

# Load the test set and predict
_, _, _, _, _, _, test_c, test_r, _ = load_data(ratio=0.05)
y_pred = model.predict(x=[test_c[:, :], test_r[:, :]])

# Save the prediction
pred_filename = model_filename[:-4] + 'pred.pkl'
with open(str(model_dir / pred_filename), 'wb') as f:
    pickle.dump(y_pred, f)
