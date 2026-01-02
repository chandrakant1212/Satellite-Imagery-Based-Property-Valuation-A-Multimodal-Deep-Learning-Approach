import os
import torch
import pandas as pd
import numpy as np
import pickle
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "outputs" if os.path.exists("outputs") else "../outputs"
DATA_DIR = "data/raw" if os.path.exists("data/raw") else "../data/raw"
IMAGE_DIR = "data/images" if os.path.exists("data/images") else "../data/images"
# Helper to handle running from scripts/ or root
if not os.path.exists(OUTPUT_DIR):
    # Try absolute path based on user workspace if relative fails completely
    OUTPUT_DIR = r"c:\Users\ENG CHANDRAKANT\jupyter\ml projects\cdc\satellite-property-valuation\outputs"
    DATA_DIR = r"c:\Users\ENG CHANDRAKANT\jupyter\ml projects\cdc\satellite-property-valuation\data\raw"
    IMAGE_DIR = r"c:\Users\ENG CHANDRAKANT\jupyter\ml projects\cdc\satellite-property-valuation\data\images"

MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")
ZIPCODE_MAP_PATH = os.path.join(OUTPUT_DIR, "zipcode_map.pkl")
TARGET_NORM_PATH = os.path.join(OUTPUT_DIR, "target_norm.json")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "final_submission.csv")

print(f"Using device: {DEVICE}")

# --- Model Definition ---
class FusionModel(nn.Module):
    def __init__(self, num_tabular_features):
        super(FusionModel, self).__init__()
        # Image Branch
        self.resnet = models.resnet18(weights=None) 
        self.resnet.fc = nn.Identity() 
        
        # Tabular Branch
        self.mlp = nn.Sequential(
            nn.Linear(num_tabular_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # Fusion
        self.head = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
    def forward(self, img, tab):
        img_x = self.resnet(img)
        tab_x = self.mlp(tab)
        combined = torch.cat((img_x, tab_x), dim=1)
        return self.head(combined).squeeze()

# --- Dataset ---
class PropertyTestDataset(Dataset):
    def __init__(self, df, image_dir, transform, feature_cols):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.feature_cols = feature_cols
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = os.path.join(self.image_dir, f"{int(row['id'])}.png")
        try:
            image = Image.open(img_name).convert('RGB')
        except:
            # Create black placeholder if image missing
            image = Image.new('RGB', (224, 224), color='black')
            
        if self.transform:
            image = self.transform(image)
            
        tabular = torch.tensor(row[self.feature_cols].values.astype(np.float32))
        return image, tabular, row['id']

def main():
    print("--- Starting Prediction Pipeline ---")
    
    # 1. Load Artifacts
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first using notebooks/model_training.ipynb")
    
    print("Loading preprocessing artifacts...")
    with open(ZIPCODE_MAP_PATH, 'rb') as f:
        zipcode_map = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(TARGET_NORM_PATH, 'r') as f:
        target_stats = json.load(f)
        y_mean = target_stats['mean']
        y_std = target_stats['std']

    # 2. Load Data
    test_path = os.path.join(DATA_DIR, "test.csv")
    print(f"Loading test data from {test_path}...")
    test_df = pd.read_csv(test_path)

    # 3. Feature Engineering (Must match training)
    test_df['date'] = pd.to_datetime(test_df['date'])
    test_df['year'] = test_df['date'].dt.year
    test_df['month'] = test_df['date'].dt.month
    test_df['house_age'] = test_df['year'] - test_df['yr_built']
    
    # Zipcode Mapping (Handle unseen with global mean)
    test_df['zipcode_mean'] = test_df['zipcode'].map(zipcode_map).fillna(y_mean)

    feature_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                    'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 
                    'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 
                    'house_age', 'zipcode_mean']

    # Scaling
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # 4. Prepare DataLoader
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_ds = PropertyTestDataset(test_df, IMAGE_DIR, transform_test, feature_cols)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 5. Load Model
    print("Loading model weights...")
    model = FusionModel(num_tabular_features=len(feature_cols)).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # 6. Inference Loop
    print("Running batch inference...")
    predictions = []
    ids = []

    with torch.no_grad():
        for imgs, tabs, batch_ids in test_loader:
            imgs = imgs.to(DEVICE)
            tabs = tabs.to(DEVICE)
            
            outputs = model(imgs, tabs)
            
            # Inverse Transform
            # Model output -> Scaled Log Price
            # 1. Unscale
            scaled_preds = outputs.cpu().numpy()
            log_preds = scaled_preds * y_std + y_mean
            
            # 2. Expm1
            real_preds = np.expm1(log_preds)
            
            predictions.extend(real_preds)
            ids.extend(batch_ids.numpy())

    # 7. Save Submission
    sub_df = pd.DataFrame({'id': ids, 'predicted_price': predictions}) # Problem statement requested 'predicted_price' header or similar? 
    # Problem statement says: "Format: id, predicted_price"
    
    sub_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Prediction file saved to {OUTPUT_PATH}")
    print(f"Total samples processed: {len(sub_df)}")

if __name__ == "__main__":
    main()
