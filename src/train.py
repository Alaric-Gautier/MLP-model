# import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from data_loader import PneumoniaDataset
from model import MLP

# Configuration pour Qt pour éviter les problèmes avec Wayland
# os.environ['QT_QPA_PLATFORM'] = 'xcb'


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience):
    model.train()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {
            epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}"
        )

        # Check for early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

    # Tracer les courbes d'apprentissage
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.show()

    return train_losses, val_losses


if __name__ == "__main__":
    DATA_PATH_TRAIN = '../resources/data/train'
    CLASS_NAME = ['NORMAL', 'PNEUMONIA']
    BATCH_SIZE = 32
    NUM_EPOCHS = 50  # Augmenté pour permettre l'arrêt anticipé
    PATIENCE = 5  # Nombre d'époques à attendre avant l'arrêt anticipé

    dataset = PneumoniaDataset(DATA_PATH_TRAIN, CLASS_NAME)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    sample_img, _ = train_dataset[0]
    print(f"Sample image shape: {sample_img.shape}")
    input_size = sample_img.shape[0]

    hidden_units = (128, 64, 32)
    num_classes = len(CLASS_NAME)

    model = MLP(input_size, hidden_units, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, PATIENCE)

    # Charger le meilleur modèle
    model.load_state_dict(torch.load('best_model.pth'))

    # Sauvegarder le modèle final
    torch.save(model.state_dict(), 'pneumonia_mlp_model.pth')
