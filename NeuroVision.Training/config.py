# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1

# Save path for models
MODEL_SAVE_PATH = "./Models/{}-model.pth"

# device
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
