from models import BasicCNN, DeepCNN, SimpleMLP
from train import train_model
from utils import plot_history

def main():
    models_dict = {
        'basiccnn': BasicCNN,
        'deepcnn': DeepCNN,
        'simplemlp': SimpleMLP,
    }
    
    print("Available models:")
    for key in models_dict:
        print(f"- {key}")

    user_input = input("Enter the name of the model you want to use: ").strip().lower()
    if user_input not in models_dict:
        print("Invalid model name entered. Defaulting to BasicCNN.")
        model = BasicCNN()
    else:
        model = models_dict[user_input]()

    model_name = model.__class__.__name__
    print(f"\nTraining {model_name}...")

    history, test_acc, test_loss = train_model(model, model_name)
    
    print(f"\nFinal Results for {model_name}:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    plot_history(history, model_name)

if __name__ == '__main__':
    main()
