from models import create_simple_cnn
from train import train_model
from utils import plot_history

def main():
    model_name = "SimpleCNN"
    model = create_simple_cnn()
    history, test_acc, test_loss = train_model(model, model_name)
    
    print(f"\nFinal Results for {model_name}:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    plot_history(history, model_name)

if __name__ == '__main__':
    main()