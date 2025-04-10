import torch
import torch.nn as nn
import torch.optim as optim
from config import EPOCHS, MODEL_SAVE_PATH, device
from data import load_data
import os
import torch.onnx

def train_model(model, model_name):
    train_loader, val_loader, test_loader = load_data()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    model.to(device)
    
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
    # Evaluation on the test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = correct / total
    print(f"Model {model_name} - Test Accuracy: {test_accuracy:.4f}")
    print(f"Model {model_name} - Test Loss: {avg_test_loss:.4f}")
    
    # Model saving (PyTorch format)
    save_path = MODEL_SAVE_PATH.format(model_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model {model_name} saved under: {save_path}")
    
    # === ONNX Export ===
    onnx_export_path = os.path.splitext(save_path)[0] + ".onnx"
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    torch.onnx.export(
        model, dummy_input, onnx_export_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )
    print(f"Model {model_name} exported to ONNX at: {onnx_export_path}")

    return history, test_accuracy, avg_test_loss