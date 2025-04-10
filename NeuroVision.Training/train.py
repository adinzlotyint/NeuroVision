from data import load_data
from config import EPOCHS, BATCH_SIZE, VALIDATION_SPLIT, MODEL_SAVE_PATH

def train_model(model, model_name):
    (x_train, y_train), (x_test, y_test) = load_data()
    
    history = model.fit(x_train, y_train, 
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_split=VALIDATION_SPLIT,
                        verbose=2)
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Model {model_name} - Test Accuracy: {test_acc:.4f}")
    print(f"Model {model_name} - Test Loss: {test_loss:.4f}")

    save_path = MODEL_SAVE_PATH.format(model_name)
    model.save(save_path)
    print(f"Model {model_name} saved under: {save_path}")
    
    return history, test_acc, test_loss