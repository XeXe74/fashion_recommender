# Fashion Recommendation System
## Author: Mario Tuset
Final Project Intelligent Systems Engineering, 2026

This project implements a complete fashion recommendation system that takes an outfit photo as input, detects individual clothes using Yolov8 and recommends visually similar items from a fashion dataset using embedding with OpenAI's CLIP model and Information Retrieval techniques.

### Key Components:
- **Cloth Detection with Yolov8:** Yolov8 fine-tuned on fashion data is used to detect and crop the clothing items from the input photo, also providing confidence scores.
- **Deep Visual Embeddings with CLIP:** OpenAI's CLIP model is used to generate deep visual embeddings for both the detected clothing items and the fashion dataset, enabling semantic similarity comparisons.
- **Constraint Satisfaction Problem (CSP) Formulation:** Supports natural language constraints provided by prompts where the user can specify budget and style. There are two types of constraints: hard constraints like budget and soft constraints penalized by TF-IDF scoring like style.
- **Hybrid Scoring Mechanism:** Uses a calibrated weighting factor (α = 0.7) to effectively balance the visual similarity from the input image with the textual relevance from the user's prompt, guaranteeing true multimodal behavior.
- **Interactive Web Interface:** Built with Gradio to provide an intuitive and easy-to-use interface where the user can upload photos, type prompts and see the recommended outfits alongside its scores.

### Project Pipeline:
The core pipeline consists of the following steps:
1. **Cloth Detection (detector.py):** The input outfit photo is processed through the Yolov8 model which separates all detected clothes and represents bounding boxes with a confidence score above 0.4 that are cropped and saved for later usage.
2. **Embedding Generation (embedder.py):** Each crop is passed through the CLIP model to get the embedding vector which contains the visual features of the clothing and then is compared against the Polyvore dataset's pre-computed embeddings using the cosine similarity to retrieve the top candidates.
3. **Scoring and Combinations (recommender.py):** Candidates are scored through a weighted formula: `Score = α * visual_score + (1-α) * text_score`. A fixed weighting factor of α = 0.7 is applied to guarantee a robust balance between the visual anchor and the textual constraints. Then, the system formulates a CSP to generate valid outfit combinations that strictly respect the maximum or minimum budget specified by the user.
4. **Visualization (app.py/visualizer.py):** The top 3 outfit combinations are sorted by their final score and displayed to the user with the final score, full price and individual cloth prices through a Gradio interface or through terminal depending on the executed file.

## Project Structure

```text
Proyecto Final/
├── data/                       # Datasets and input/output folders
│   ├── input_outfits/          # Test images for detection
│   ├── output/                 # Generated files
│   │   ├── crops/              # Intermediate cropped garments from YOLOv8
│   └── polyvore_outfits/data/  # Indexed Polyvore dataset
├── runs/                       # YOLOv8 training runs and weights
│── deepfashion2/               # Dataset used for training the Yolov8 model
├── output/                     # Datasets and input/output folders
│   ├── catalog_embeddings.pkl # Embeddings for the Polyvore dataset
│   ├── recommendations.png    # Output visualization
# --- Core Pipeline (End-to-End System) ---
├── app.py                      # Gradio Web UI
├── main.py                     # CLI version of the pipeline for terminal execution
├── detector.py                 # YOLOv8 object detection and image cropping
├── embedder.py                 # CLIP embeddings and visual cosine similarity matching
├── recommender.py              # CSP logic, textual TF-IDF scoring, and outfit ranking
├── visualizer.py               # Dataset indexing and Matplotlib visualization (for CLI)
│
# --- Data Preparation & Model Training ---
├── train_yolo.py               # Script used to train the YOLOv8 custom clothing model
├── preparing_outfits.py        # Dataset preprocessing and formatting
├── index_dataset.py            # Utility to index the catalog data
├── add_prices.py               # Utility to simulate item prices in the catalog
├── polyvore_data_type.py       # Data definitions for the Polyvore dataset
├── embedder_resnet.py          # Alternative embedding model with ResNet
│
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## Installation

To set up the project, follow these steps:
1. **Clone the Repository:**
    ```bash
   git clone <your-repository-url>
   cd <repository-folder>
   ```
2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure the Polyvore dataset is extracted in `data/polyvore_outfits/data` and the YOLOv8 weights are placed in `runs/detect/runs/fashion_detector/weights/best.pt`.


## Usage

### Web Interface:
To use the Gradio Interface to the best experience, run:
```bash
python app.py
```
This will launch a local server where photos and prompts can be uploaded.

### Command Line Interface:
For a terminal-based execution, run:
```bash
python main.py
```
To select the input image, it must be specified in code, but the prompt is asked by terminal. As a result, the recommended outfits and their scores will appear in a .png and saved in output