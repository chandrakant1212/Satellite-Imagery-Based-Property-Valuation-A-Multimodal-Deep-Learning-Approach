# Satellite Imagery-Based Property Valuation

## Overview
This project enhances property valuation by integrating "curb appeal" and environmental context from satellite imagery with traditional tabular data. We utilize a **Multimodal Deep Learning Pipeline** that fuses a **ResNet-18 CNN** (Visual Model) with a **Multi-Layer Perceptron** (Tabular Model) to predict property prices with high accuracy (R² ~0.8250).

## Narrative & Methodology

### 1. Problem Statement
Traditional valuation models rely solely on spreadsheet numbers—bedrooms, sqft, year built. They miss a critical factor: **Curb Appeal**. A house next to a lush park is worth more than one next to a highway, even if their stats are identical.

### 2. Solution Overview
We built a **Multimodal Valuation Pipeline** that "sees" the neighborhood. By combining satellite imagery with tabular data, we capture the environmental context that drives value.

### 3. Technical Architecture
Our **Hybrid Fusion Architecture** (PyTorch) consists of:
1.  **Visual Branch**: **ResNet-18** (Pre-trained on ImageNet) to extract 512-dimensional visual embeddings from satellite images. We use **Transfer Learning** by initially freezing the backbone and then fine-tuning it.
2.  **Tabular Branch**: A **Multi-Layer Perceptron (MLP)** processes numeric features, including a powerful **Target Encoded Zipcode** feature (`zipcode_mean`) that injects neighborhood price baselines.
3.  **Fusion Layer**: Concatenates visual and structural embeddings to predict the final price.

### 4. Key Challenges & Solutions
*   **Small Data for Deep Learning**: CNNs can overfit on small datasets. We solved this by using **Target Encoding Injection** (`zipcode_mean`) to give the model a strong baseline, allowing the CNN to focus on learning residual visual signals.
*   **Explainability**: We implemented **Grad-CAM**, which overlays a heatmap on the satellite image to show *why* the model predicted a certain price (e.g., highlighting green spaces or water bodies).

### 5. Performance Impact
The final model achieves an **R² score of 0.8250**, significantly outperforming the tabular-only baseline (0.7953).

## Project Structure
- `data/`: Raw CSVs and images.
- `notebooks/`: 
    - `preprocessing.ipynb`: Data cleaning & Target Encoding.
    - `model_training.ipynb`: Fusion Model Training.
    - `visualization.ipynb`: Grad-CAM Explainability.
- `data_fetcher.py`: Downloads satellite images.
- `generate_predictions.py`: Generates inference on test data.
- `outputs/`: Model artifacts & predictions.

## Setup & Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Data Fetching
(Optional) Download new satellite images:
```bash
python data_fetcher.py --limit 1000
```

### 3. Training & Analysis
Run the notebooks in order:
1.  `notebooks/preprocessing.ipynb`: Prepares data.
2.  `notebooks/model_training.ipynb`: Trains the Multimodal Model.
3.  `notebooks/visualization.ipynb`: Visualizes model attention.

### 4. Generate Predictions
Create `final_submission.csv`:
```bash
python generate_predictions.py
```
