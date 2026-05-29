import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)

# ── WHAT THIS MODULE DOES ─────────────────────────────────────────────────────
# Classifies wafer map defect patterns from the WM-811K dataset
# Each wafer map is a 2D image showing which dies passed (1) or failed (0)
# Defect patterns tell process engineers WHAT went wrong:
#   - Center: contamination or plasma issue at center
#   - Edge-ring: edge effects from deposition equipment
#   - Scratch: handling damage
#   - Donut: focus/exposure issue in lithography
# We train a CNN and add Grad-CAM to show WHERE the network looks

print("Loading WM-811K dataset (this takes ~60 seconds for 2GB file)...")

# ── 1. LOAD AND FILTER DATA ───────────────────────────────────────────────────
df = pd.read_pickle('data/LSWMD.pkl')
print(f"Total wafer maps: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Keep only labeled samples (80% are unlabeled)
df = df.dropna(subset=['failureType'])
df = df[df['failureType'] != 0]
# Flatten nested array labels to plain strings
df['failureType'] = df['failureType'].apply(
    lambda x: x[0][0] if isinstance(x, np.ndarray) and len(x) > 0 
              and isinstance(x[0], np.ndarray) else str(x)
)
# Remove any remaining non-string or empty values
df = df[df['failureType'].apply(lambda x: isinstance(x, str) and len(x) > 0)]
df = df[df['failureType'] != 'none']
print(f"Labeled wafers: {len(df)}")
print(f"Defect types:\n{df['failureType'].value_counts()}")

# ── 2. PREPARE IMAGES ─────────────────────────────────────────────────────────
# Each wafer map is a 2D numpy array of varying size
# We resize everything to 32x32 for the CNN

IMG_SIZE = 32

# Defect class names
DEFECT_NAMES = {
    'Center': 'Center', 'Donut': 'Donut', 'Edge-Loc': 'Edge-Loc',
    'Edge-Ring': 'Edge-Ring', 'Loc': 'Loc', 'Near-full': 'Near-Full',
    'Random': 'Random', 'Scratch': 'Scratch', 'none': 'None'
}

def preprocess_wafer(wafer_map):
    """Resize wafer map to IMG_SIZE x IMG_SIZE and normalize."""
    if wafer_map is None or len(wafer_map) == 0:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    
    # Convert to float and normalize to 0-1
    wm = np.array(wafer_map, dtype=np.float32)
    
    # Resize using PIL
    img = Image.fromarray((wm * 127).astype(np.uint8))
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    result = np.array(img, dtype=np.float32) / 127.0
    return result

print("Preprocessing wafer maps...")
# Use a balanced subset: 300 per class max for speed
samples_per_class = 300
balanced_dfs = []
for ftype in df['failureType'].unique():
    class_df = df[df['failureType'] == ftype]
    n = min(len(class_df), samples_per_class)
    balanced_dfs.append(class_df.sample(n, random_state=42))

df_balanced = pd.concat(balanced_dfs).reset_index(drop=True)
print(f"Balanced dataset size: {len(df_balanced)}")
print(f"Class distribution:\n{df_balanced['failureType'].value_counts()}")

# Process images
X_images = np.array([preprocess_wafer(wm) 
                     for wm in df_balanced['waferMap']], dtype=np.float32)
X_images = X_images[:, np.newaxis, :, :]  # add channel dim for CNN

# Encode labels to 0-based integers
le = LabelEncoder()
y_labels = le.fit_transform(df_balanced['failureType'].values)
class_names = [DEFECT_NAMES.get(str(c), str(c)) for c in le.classes_]
print(f"Classes: {class_names}")

# ── 3. VISUALIZE SAMPLE WAFER MAPS ────────────────────────────────────────────
print("Saving sample wafer map visualization...")
n_classes = len(class_names)
fig, axes = plt.subplots(1, n_classes, figsize=(2.5*n_classes, 3))
if n_classes == 1:
    axes = [axes]

for idx, cls in enumerate(le.classes_):
    mask = df_balanced['failureType'] == cls
    sample = df_balanced[mask].iloc[0]['waferMap']
    wm = np.array(sample, dtype=np.float32)
    axes[idx].imshow(wm, cmap='RdYlGn', vmin=0, vmax=2)
    axes[idx].set_title(DEFECT_NAMES.get(cls, f'Type_{cls}'), 
                        fontsize=9, fontweight='bold')
    axes[idx].axis('off')

plt.suptitle('WM-811K Wafer Map Defect Patterns\n'
             'Green=Pass Die, Red=Fail Die, Yellow=Untested',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/module3_wafer_samples.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/module3_wafer_samples.png")

# ── 4. CNN MODEL ──────────────────────────────────────────────────────────────
# Simple but effective CNN for wafer map classification
# Three conv layers + pooling + fully connected

class WaferCNN(nn.Module):
    def __init__(self, num_classes):
        super(WaferCNN, self).__init__()
        # Conv layers extract spatial features (edge patterns, rings, etc.)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        
        # Calculate size after three pooling layers: 32 -> 16 -> 8 -> 4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Store gradients for Grad-CAM
        self.gradients = None
        self.activations = None
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def forward(self, x, get_cam=False):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        
        if get_cam:
            # Register hook to capture gradients for Grad-CAM
            self.activations = x
            x.register_hook(self.activations_hook)
        
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ── 5. DATASET AND TRAINING ───────────────────────────────────────────────────
class WaferDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

X_train, X_test, y_train, y_test = train_test_split(
    X_images, y_labels, test_size=0.2, random_state=42, stratify=y_labels
)

train_loader = DataLoader(WaferDataset(X_train, y_train), 
                          batch_size=32, shuffle=True)
test_loader  = DataLoader(WaferDataset(X_test, y_test), 
                          batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

model = WaferCNN(num_classes=len(class_names)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("Training CNN (15 epochs)...")
train_losses, train_accs = [], []

for epoch in range(15):
    model.train()
    correct, total, loss_sum = 0, 0, 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item()
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()
    
    acc = 100 * correct / total
    train_losses.append(loss_sum / len(train_loader))
    train_accs.append(acc)
    
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1:2d}/15 | Loss: {loss_sum/len(train_loader):.4f} | Acc: {acc:.1f}%")

# ── 6. EVALUATE ───────────────────────────────────────────────────────────────
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.numpy())

print("\n── Classification Report ──")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(class_names, fontsize=9)
ax.set_xlabel('Predicted Defect Type')
ax.set_ylabel('Actual Defect Type')
ax.set_title('Wafer Defect Classification — Confusion Matrix\nCNN on WM-811K Dataset',
             fontweight='bold')
for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax.text(j, i, cm[i,j], ha='center', va='center',
                color='white' if cm[i,j] > cm.max()*0.5 else 'black',
                fontsize=8, fontweight='bold')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('outputs/module3_confusion_matrix.png', dpi=150)
plt.close()
print("Saved: outputs/module3_confusion_matrix.png")

# ── 7. GRAD-CAM VISUALIZATION ─────────────────────────────────────────────────
# Grad-CAM shows WHERE in the wafer map the CNN is looking
# This is the key insight: it spatially localizes the defect
# A process engineer can see "the model focuses on the edge ring"

print("Generating Grad-CAM visualizations...")

def get_gradcam(model, image_tensor, target_class, device):
    """Generate Grad-CAM heatmap for a single image."""
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    image_tensor.requires_grad_(True)
    
    output = model(image_tensor, get_cam=True)
    
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0][target_class] = 1
    output.backward(gradient=one_hot, retain_graph=True)
    
    if model.gradients is None or model.activations is None:
        return np.zeros((IMG_SIZE, IMG_SIZE))
    
    gradients = model.gradients.detach().cpu().numpy()[0]
    activations = model.activations.detach().cpu().numpy()[0]
    
    weights = gradients.mean(axis=(1, 2))
    cam = np.zeros(activations.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * activations[i]
    
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR
        ), dtype=np.float32
    ) / 255.0
    
    return cam_resized

# Show Grad-CAM for one sample per class
n_show = min(6, len(class_names))
fig, axes = plt.subplots(3, n_show, figsize=(2.5*n_show, 8))

for col_idx in range(n_show):
    cls_idx = col_idx
    cls_name = class_names[cls_idx]
    
    # Find a correctly classified sample
    mask = np.array(all_labels) == cls_idx
    test_indices = np.where(mask)[0]
    
    if len(test_indices) == 0:
        continue
    
    sample_idx = test_indices[0]
    image = torch.FloatTensor(X_test[sample_idx])
    true_label = y_test[sample_idx]
    
    # Get prediction
    with torch.no_grad():
        pred = model(image.unsqueeze(0).to(device)).argmax().item()
    
    # Get Grad-CAM
    cam = get_gradcam(model, image, pred, device)
    
    # Row 0: original wafer map
    axes[0, col_idx].imshow(X_test[sample_idx, 0], cmap='RdYlGn', vmin=0, vmax=1)
    axes[0, col_idx].set_title(cls_name, fontsize=9, fontweight='bold')
    axes[0, col_idx].axis('off')
    
    # Row 1: Grad-CAM heatmap
    axes[1, col_idx].imshow(cam, cmap='jet', vmin=0, vmax=1)
    axes[1, col_idx].set_title('Attention', fontsize=8)
    axes[1, col_idx].axis('off')
    
    # Row 2: overlay
    wafer_rgb = plt.cm.RdYlGn(X_test[sample_idx, 0])[:, :, :3]
    cam_rgb = plt.cm.jet(cam)[:, :, :3]
    overlay = 0.6 * wafer_rgb + 0.4 * cam_rgb
    axes[2, col_idx].imshow(overlay)
    pred_name = class_names[pred] if pred < len(class_names) else 'Unknown'
    color = 'green' if pred == true_label else 'red'
    axes[2, col_idx].set_title(f'Pred: {pred_name}', fontsize=8, color=color)
    axes[2, col_idx].axis('off')

axes[0, 0].set_ylabel('Wafer Map', fontsize=9)
axes[1, 0].set_ylabel('Grad-CAM', fontsize=9)
axes[2, 0].set_ylabel('Overlay', fontsize=9)

plt.suptitle('Grad-CAM: Where the CNN Looks to Classify Defects\n'
             'Row 1: Wafer Map | Row 2: Attention Heatmap | Row 3: Overlay',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/module3_gradcam.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/module3_gradcam.png")

# Training curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(train_losses, 'b-o', markersize=4)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Training Loss'); ax1.grid(True, alpha=0.3)

ax2.plot(train_accs, 'g-o', markersize=4)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training Accuracy'); ax2.grid(True, alpha=0.3)

plt.suptitle('CNN Training History — Wafer Defect Classifier', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/module3_training_curve.png', dpi=150)
plt.close()
print("Saved: outputs/module3_training_curve.png")

print("\nModule 3 complete.")