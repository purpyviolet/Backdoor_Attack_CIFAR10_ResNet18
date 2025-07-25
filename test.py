import torch
import torch.nn as nn
from utils.readData_attack import read_dataset_test_A, read_dataset_test_B, read_dataset
from resnet18 import ResNet18

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_class = 10
batch_size = 100

# Load the pre-trained model
model = ResNet18()
model.load_state_dict(torch.load('AttackResult/model-a.pth')) # change this into model.pth to obtain the original model test results
model = model.to(device)

# Define a function to evaluate the model on a given test loader
def evaluate_model(test_loader, description):
    total_sample = 0
    right_sample = 0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)  # Forward pass
            _, pred = torch.max(output, 1)  # Get the predicted class
            correct_tensor = pred.eq(target.data.view_as(pred))  # Compare predictions with true labels
            total_sample += batch_size
            right_sample += correct_tensor.sum().item()  # Sum the correct predictions
    accuracy = 100 * right_sample / total_sample
    print(f"{description} Accuracy: {accuracy:.2f}%")

# Test on Trigger A dataset
test_loader_A = read_dataset_test_A(batch_size=batch_size, pic_path='data')
evaluate_model(test_loader_A, "Trigger A Test Set")

# Test on Trigger B dataset
test_loader_B = read_dataset_test_B(batch_size=batch_size, pic_path='data')
evaluate_model(test_loader_B, "Trigger B Test Set")

# Test on Clean dataset
_, _, test_loader_clean = read_dataset(batch_size=batch_size, pic_path='data')
evaluate_model(test_loader_clean, "Clean Test Set")