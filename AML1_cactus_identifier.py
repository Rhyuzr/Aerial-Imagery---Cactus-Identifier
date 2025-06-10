import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from tqdm import tqdm

# Load the labels
try:
    labels_df = pd.read_csv('train.csv')
    print("Loaded train.csv from current directory")
except FileNotFoundError:
    try:
        labels_df = pd.read_csv('train/train.csv')
        print("Loaded train.csv from train/ directory")
    except FileNotFoundError:
        print("Could not find train.csv, please check the file location")
        raise

# Display class distribution
print("Class distribution:")
print(labels_df['has_cactus'].value_counts())
print(f"Percentage of cactus images: {100 * labels_df['has_cactus'].mean():.2f}%")

# Determine the correct train directory structure
train_dir = ''
if os.path.exists('train/train'):
    train_dir = 'train/train'
    print("Using 'train/train' directory for images")
elif os.path.exists('train'):
    # Check if the first image exists in this directory
    if len(labels_df) > 0 and os.path.exists(os.path.join('train', labels_df.iloc[0, 0])):
        train_dir = 'train'
        print("Using 'train' directory for images")
    else:
        print("Warning: 'train' directory exists but images not found there")
        train_dir = '.'
else:
    print("Warning: Could not find train directory. Using current directory.")
    train_dir = '.'

sample_images = [f for f in os.listdir(train_dir)[:5] if f.endswith('.jpg') or f.endswith('.png')]
if sample_images:
    fig, axes = plt.subplots(1, min(5, len(sample_images)), figsize=(15, 3))
    if len(sample_images) == 1:
        axes = [axes]  # Make axes iterable for single image case
    for img_path, ax in zip(sample_images, axes):
        try:
            img = Image.open(os.path.join(train_dir, img_path))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'Cactus: {labels_df.loc[labels_df["id"] == img_path, "has_cactus"].values[0]}')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    plt.show()
else:
    print("No image files found in the directory.")

# Print the first few rows of the dataframe to understand structure
print("First few rows of the dataset:")
print(labels_df.head())

# Define a custom dataset
class CactusDataset(Dataset):
    def __init__(self, labels_df, img_dir, transform=None):
        self.labels_df = labels_df
        self.img_dir = img_dir
        self.transform = transform
        # Debugging: print sample paths
        if len(labels_df) > 0:
            sample_id = labels_df.iloc[0, 0]
            print(f"Sample image path: {os.path.join(img_dir, sample_id)}")
            print(f"This file exists: {os.path.exists(os.path.join(img_dir, sample_id))}")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        try:
            img_id = self.labels_df.iloc[idx, 0]
            img_path = os.path.join(self.img_dir, img_id)
            
            # Debug: if file doesn't exist, look for alternatives
            if not os.path.exists(img_path):
                print(f"Warning: Image not found at {img_path}")
                # Try listing some files in the directory to see what's available
                if os.path.exists(self.img_dir):
                    some_files = os.listdir(self.img_dir)[:5]
                    print(f"Some files in directory: {some_files}")
            
            image = Image.open(img_path)
            label = self.labels_df.iloc[idx, 1]
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            # Return a placeholder image and the label
            # For training to continue without crashing
            dummy_img = torch.zeros((3, 32, 32)) if self.transform else Image.new('RGB', (32, 32))
            return dummy_img, self.labels_df.iloc[idx, 1]

# Define transformations with augmentation for training
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RGB images
])

# Simpler transform for validation and testing
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RGB images
])

# Create dataset
dataset = CactusDataset(labels_df, train_dir, transform=train_transform)

# Split the dataset into train (60%), validation (20%), test (20%)
train_df, temp_df = train_test_split(labels_df, test_size=0.4, stratify=labels_df['has_cactus'], random_state=42)
val_df, test_internal_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['has_cactus'], random_state=42)

# Create data loaders with appropriate transforms
train_dataset = CactusDataset(train_df, train_dir, transform=train_transform)
val_dataset = CactusDataset(val_df, train_dir, transform=eval_transform)
test_internal_dataset = CactusDataset(test_internal_df, train_dir, transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_internal_loader = DataLoader(test_internal_dataset, batch_size=32, shuffle=False)

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(-1, 64 * 4 * 4)  # Adjusted based on output size after convolutions and pooling
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Add learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# Training loop with validation after each epoch
num_epochs = 15  # Using 15 epochs as requested
best_val_f1 = 0.0
best_model = None
patience = 7  # For early stopping
no_improve_epochs = 0

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Validation phase
    model.eval()
    val_preds = []
    val_labels = []
    val_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            val_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
    
    # Calculate validation metrics
    val_f1 = f1_score(val_labels, val_preds)
    
    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model = model.state_dict().copy()
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
    
    # Update learning rate based on F1 score
    scheduler.step(val_f1)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Val F1: {val_f1:.4f}, LR: {current_lr:.6f}')
    
    # Early stopping
    if no_improve_epochs >= patience:
        print(f'Early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)')
        break

# Create a function for evaluation and visualization
def evaluate_model(model, data_loader, dataset_name='Test'):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=f'Evaluating on {dataset_name} set'):
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
    
    # Calculate metrics
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    f1 = f1_score(all_labels, all_preds)
    
    # Print classification report
    print(f"\n{dataset_name} Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['No Cactus', 'Cactus']))
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{dataset_name} Confusion Matrix')
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['No Cactus', 'Cactus'])
    plt.yticks(tick_marks, ['No Cactus', 'Cactus'])
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'confusion_matrix_{dataset_name.lower()}.png')
    plt.show()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_curve_{dataset_name.lower()}.png')
    plt.show()
    
    return {
        'accuracy': acc,
        'f1_score': f1,
        'auc': roc_auc,
        'predictions': all_preds,
        'probabilities': all_probs,
        'true_labels': all_labels
    }

# Load best model for final evaluation
if best_model is not None:
    model.load_state_dict(best_model)
    # Save the best model to file
    torch.save(model.state_dict(), 'cactus_classifier_best_model.pth')
    print("Best model saved to cactus_classifier_best_model.pth")
model.eval()

# Evaluate the model on internal test set
print("\n=== Evaluation on Internal Test Set ===\n")
test_results = evaluate_model(model, test_internal_loader, 'Internal Test')

# Print overall test metrics
print(f"\nTest Accuracy: {test_results['accuracy']:.4f}")
print(f"Test F1 Score: {test_results['f1_score']:.4f}")
print(f"Test AUC: {test_results['auc']:.4f}")

# Now we don't need to save model again as we already saved the best model

# =============================================
# Random Forest Classifier for comparison
# =============================================
print("\n=== Training and Evaluating Random Forest Classifier ===")

# Function to convert image to feature vector (flatten RGB channels)
def extract_features(loader):
    all_features = []
    all_labels = []
    for images, labels in tqdm(loader, desc="Extracting features"):
        # Flatten images: each 32x32 RGB image becomes a 3072-length vector
        features = images.view(images.size(0), -1).cpu().numpy()
        all_features.append(features)
        all_labels.append(labels.cpu().numpy())
    
    return np.vstack(all_features), np.concatenate(all_labels)

# Extract features from train, validation and test sets
print("Extracting features for Random Forest...")
X_train, y_train = extract_features(train_loader)
X_val, y_val = extract_features(val_loader)
X_test, y_test = extract_features(test_internal_loader)

print(f"Feature shapes - Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
print("Training Random Forest...")
rf_model.fit(X_train, y_train)

# Evaluate on validation set
val_rf_preds = rf_model.predict(X_val)
val_rf_probs = rf_model.predict_proba(X_val)[:, 1]

val_rf_accuracy = accuracy_score(y_val, val_rf_preds)
val_rf_f1 = f1_score(y_val, val_rf_preds)

# Evaluate on test set
test_rf_preds = rf_model.predict(X_test)
test_rf_probs = rf_model.predict_proba(X_test)[:, 1]

test_rf_accuracy = accuracy_score(y_test, test_rf_preds)
test_rf_f1 = f1_score(y_test, test_rf_preds)

# Print RF results
print("\nRandom Forest Results:")
print(f"Validation Accuracy: {val_rf_accuracy:.4f}")
print(f"Validation F1 Score: {val_rf_f1:.4f}")
print(f"Test Accuracy: {test_rf_accuracy:.4f}")
print(f"Test F1 Score: {test_rf_f1:.4f}")

# Print classification report
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, test_rf_preds, target_names=['No Cactus', 'Cactus']))

# Plot confusion matrix for RF
cm_rf = confusion_matrix(y_test, test_rf_preds)
plt.figure(figsize=(8, 6))
plt.imshow(cm_rf, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Random Forest Confusion Matrix')
plt.colorbar()
tick_marks = [0, 1]
plt.xticks(tick_marks, ['No Cactus', 'Cactus'])
plt.yticks(tick_marks, ['No Cactus', 'Cactus'])

# Add numbers to confusion matrix
thresh = cm_rf.max() / 2.
for i in range(cm_rf.shape[0]):
    for j in range(cm_rf.shape[1]):
        plt.text(j, i, format(cm_rf[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm_rf[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('rf_confusion_matrix.png')
plt.show()

# Plot ROC curve for RF
fpr_rf, tpr_rf, _ = roc_curve(y_test, test_rf_probs)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Get the CNN's ROC data for comparison
fpr_cnn = test_results['fpr'] if 'fpr' in test_results else None
tpr_cnn = test_results['tpr'] if 'tpr' in test_results else None
roc_auc_cnn = test_results['auc']

# Plot both ROC curves for comparison
plt.figure(figsize=(10, 8))
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.3f})')

# Only add CNN curve if we have the data
if fpr_cnn is not None and tpr_cnn is not None:
    plt.plot(fpr_cnn, tpr_cnn, color='darkgreen', lw=2, label=f'CNN (AUC = {roc_auc_cnn:.3f})')
else:
    # Add a note about CNN AUC
    plt.annotate(f'CNN AUC = {roc_auc_cnn:.3f}', xy=(0.5, 0.1), xycoords='axes fraction')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparison of ROC Curves: CNN vs Random Forest')
plt.legend(loc="lower right")
plt.savefig('roc_comparison.png')
plt.show()

# Comparison table
print("\n=== Model Comparison ===")
print(f"{'Model':<15} {'Test Accuracy':<15} {'Test F1 Score':<15} {'Test AUC':<15}")
print("-" * 60)
print(f"{'CNN':<15} {test_results['accuracy']:<15.4f} {test_results['f1_score']:<15.4f} {test_results['auc']:<15.4f}")
print(f"{'Random Forest':<15} {test_rf_accuracy:<15.4f} {test_rf_f1:<15.4f} {roc_auc_rf:<15.4f}")

# Predict on test data
print("\n=== Generating Predictions for Test Set ===")

# Create a custom dataset for test images (without labels)
class TestCactusDataset(Dataset):
    def __init__(self, file_list, img_dir, transform=None):
        self.file_list = file_list
        self.img_dir = img_dir
        self.transform = transform
        print(f"TestCactusDataset initialized with {len(file_list)} files in {img_dir}")
        # Verify a few paths
        if len(file_list) > 0:
            sample_path = os.path.join(img_dir, file_list[0])
            print(f"Sample test image path: {sample_path}")
            print(f"This file exists: {os.path.exists(sample_path)}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        try:
            img_name = os.path.join(self.img_dir, self.file_list[idx])
            
            # Debugging
            if idx < 5:  # Print info for the first few images
                print(f"Loading test image {idx}: {img_name}")
                print(f"File exists: {os.path.exists(img_name)}")
                
            image = Image.open(img_name)
            if self.transform:
                image = self.transform(image)
            return image, self.file_list[idx]
        except Exception as e:
            print(f"Error loading test image at index {idx}: {e}")
            # Return a placeholder
            dummy_img = torch.zeros((3, 32, 32)) if self.transform else Image.new('RGB', (32, 32))
            return dummy_img, self.file_list[idx]

# Find the test directory and print more debug info
test_dir = ''
test_files = []

# Try different possible locations
possible_test_dirs = ['test/', 'test', './test', '../test']

for dir_path in possible_test_dirs:
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        try:
            files = os.listdir(dir_path)
            if files:  # Make sure there are files
                # Check if the files look like images
                image_files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
                if image_files:
                    test_dir = dir_path
                    test_files = image_files
                    print(f"Found {len(image_files)} image files in {dir_path}")
                    print(f"First few test files: {image_files[:5]}")
                    break
                else:
                    print(f"Directory {dir_path} exists but contains no image files")
            else:
                print(f"Directory {dir_path} exists but is empty")
        except PermissionError:
            print(f"Permission denied when trying to access {dir_path}")

if not test_files:
    print("Could not find test directory with image files. Please check the path.")

if test_files:
    # Create test dataset and loader
    test_dataset = TestCactusDataset(test_files, test_dir, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Generate predictions
    predictions = []
    image_ids = []

    model.eval()
    with torch.no_grad():
        for images, img_ids in tqdm(test_loader, desc='Predicting on test set'):
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            image_ids.extend(img_ids)

    # Create submission file
    submission_df = pd.DataFrame({'id': image_ids, 'has_cactus': predictions})
    submission_df.to_csv('submission.csv', index=False)
    print(f"\nSubmission file created with {len(submission_df)} predictions.")
    print("Sample predictions:")
    print(submission_df.head())
else:
    print("No test files found. Skipping test predictions.")
