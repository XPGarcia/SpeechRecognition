import Trainer
import matplotlib.pyplot as plt

# Build or retrain the model for 10000 records over 50 epochs
# train(n_train, model_name, previous_model_name, build=False)


def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper left")
    plt.show()


# Build model
# Trainer.train(0, "model_speech", "", build=True)

# Retrain model
for n_train in range(1, 2):
    previous_model_name = "model_speech"
    model_name = "model_speech_" + str(n_train)
    Trainer.train(n_train, model_name, previous_model_name)
