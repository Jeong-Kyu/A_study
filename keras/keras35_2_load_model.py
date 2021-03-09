from tensorflow.keras.models import load_model
model = load_model("../data/h5/save_keras35.h5")

model.summary()

# WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.