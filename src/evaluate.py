import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import PneumoniaDataset
from model import MLP

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    return all_labels, all_preds

if __name__ == "__main__":
    DATA_PATH_TEST = '../resources/data/test'
    CLASS_NAME = ['NORMAL', 'PNEUMONIA']

    test_dataset = PneumoniaDataset(DATA_PATH_TEST, CLASS_NAME)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_size = test_dataset[0][0].shape[0]
    hidden_size = 512
    num_classes = len(CLASS_NAME)

    model = MLP(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load('pneumonia_mlp_model.pth'))

    labels, preds = evaluate_model(model, test_loader)

    # Afficher le rapport de classification
    print(classification_report(labels, preds, target_names=CLASS_NAME))

    # Afficher la matrice de confusion
    conf_matrix = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAME, yticklabels=CLASS_NAME)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
