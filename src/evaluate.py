import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import PneumoniaDataset
from model import MLP  # Assurez-vous d'importer le bon modèle


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

    sample_img, _ = test_dataset[0]
    # Assurez-vous que les dimensions de l'image sont correctement obtenues
    image_shape = sample_img.shape
    print(f"Sample image shape: {image_shape}")

    # input_size = image_shape[0] * image_shape[1] * image_shape[2]
    input_size = image_shape[0]

    hidden_units = (128, 64, 32)  # Utilisez un tuple pour hidden_units
    num_classes = len(CLASS_NAME)

    model = MLP(input_size, hidden_units, num_classes)
    model.load_state_dict(torch.load('pneumonia_mlp_model.pth'))

    labels, preds = evaluate_model(model, test_loader)

    # Afficher le rapport de classification
    report = classification_report(
        labels, preds, target_names=CLASS_NAME, output_dict=True)
    print(classification_report(labels, preds, target_names=CLASS_NAME))

    # Afficher la matrice de confusion
    conf_matrix = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAME, yticklabels=CLASS_NAME)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Calcul des taux pour les graphiques ronds
    normal_precision = report['NORMAL']['precision']
    normal_recall = report['NORMAL']['recall']
    pneumonia_precision = report['PNEUMONIA']['precision']
    pneumonia_recall = report['PNEUMONIA']['recall']

    # Diagrammes circulaires pour la précision
    labels = ['Precision (NORMAL)', 'Precision (PNEUMONIA)']
    sizes = [normal_precision, pneumonia_precision]
    colors = ['#ff9999', '#66b3ff']
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=140)
    plt.title('Precision Rates')
    plt.show()

    # Diagrammes circulaires pour le rappel
    labels = ['Recall (NORMAL)', 'Recall (PNEUMONIA)']
    sizes = [normal_recall, pneumonia_recall]
    colors = ['#ff9999', '#66b3ff']
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=140)
    plt.title('Recall Rates')
    plt.show()

    # Diagrammes circulaires pour les faux positifs
    normal_fpr = report['NORMAL']['f1-score']
    pneumonia_fpr = report['PNEUMONIA']['f1-score']
    labels = ['F1 Score (NORMAL)', 'F1 Score (PNEUMONIA)']
    sizes = [normal_fpr, pneumonia_fpr]
    colors = ['#ff9999', '#66b3ff']
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=140)
    plt.title('F1 Score')
    plt.show()

    # Diagrammes circulaires pour les faux négatifs
    normal_fnr = report['NORMAL']['support']
    pneumonia_fnr = report['PNEUMONIA']['support']
    labels = ['Support (NORMAL)', 'Support (PNEUMONIA)']
    sizes = [normal_fnr, pneumonia_fnr]
    colors = ['#ff9999', '#66b3ff']
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=140)
    plt.title('Support')
    plt.show()
