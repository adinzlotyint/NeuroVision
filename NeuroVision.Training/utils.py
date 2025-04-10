import matplotlib.pyplot as plt

def plot_history(history, model_name):
    plt.figure(figsize=(12, 8))
    plt.plot(history['val_accuracy'], label=f"{model_name} - Val Accuracy")
    plt.title(f"Validation Accuracy for {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()
