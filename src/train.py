import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import PneumoniaDataset
from model import MLP


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")


if __name__ == "__main__":
    DATA_PATH_TRAIN = '../resources/data/train'
    CLASS_NAME = ['NORMAL', 'PNEUMONIA']
    BATCH_SIZE = 32
    NUM_EPOCHS = 20

    train_dataset = PneumoniaDataset(DATA_PATH_TRAIN, CLASS_NAME)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_size = train_dataset[0][0].shape[0]
    hidden_size = 512
    num_classes = len(CLASS_NAME)

    model = MLP(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS)

    model.save('pneumonia_mlp_model.pth')
