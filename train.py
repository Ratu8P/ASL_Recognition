import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from model import KeypointCNN as CNN
from data_loader import HandKeypointsDataset
import os

# data paths
train_dir = "./data/processed_dataset_split_L/train"
valid_dir = "./data/processed_dataset_split_L/valid"
test_dir = "./data/processed_dataset_split_L/test"
EPOCH = 50

#load data
train_dataset = HandKeypointsDataset(train_dir)
valid_dataset = HandKeypointsDataset(valid_dir)
test_dataset = HandKeypointsDataset(test_dir)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# obtain class mapping
classes = train_dataset.classes
num_classes = len(classes)
with open("class_mapping.txt", "w") as f:
    for idx, cls_name in enumerate(classes):
        f.write(f"{idx}: {cls_name}\n")
print("Class mapping saved to 'class_mapping.txt'.")


sample_input, _ = next(iter(train_loader))
input_dim = sample_input.shape[1]

model = CNN(num_classes=num_classes)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(torch.backends.mps.is_available())
model.to(device)

criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
    accuracy = correct / len(data_loader.dataset)
    return total_loss / len(data_loader), accuracy

# save results
train_losses = []
val_losses = []
val_accuracies = []
test_accuracies = []

# training loop
for epoch in range(EPOCH):
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # valid
    val_loss, val_accuracy = evaluate(model, valid_loader, criterion)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    # test
    _, test_accuracy = evaluate(model, test_loader, criterion)
    test_accuracies.append(test_accuracy)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # update learning rate
    scheduler.step()

# save model
model_dir = "keypoint_model_L_cnn_50.pth"
torch.save(model.state_dict(), model_dir)
print("Model saved as " + model_dir)

# draw training curve
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation and Test Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("training_curve.png")
plt.show()
