# Aerial-Imagery - Cactus-Identifier

A deep learning project for identifying cacti in aerial imagery.

## Description

This project uses deep learning techniques to automatically identify the presence of cacti in aerial images. The model is built with PyTorch and utilizes Convolutional Neural Networks (CNN) for classification.

## Prerequisites

- Python 3.8+
- PyTorch
- Pandas
- NumPy
- Pillow (PIL)
- Matplotlib
- Scikit-learn
- tqdm

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install torch pandas numpy pillow matplotlib scikit-learn tqdm
```

## Project Structure

```
.
├── train/              # Training data directory
│   └── train.csv      # Labels file
└── AML1_cactus_identifier.py  # Main script
```

## Usage

1. Ensure your training data is in the `train/` directory with the `train.csv` file.
2. Run the main script:
```bash
python AML1_cactus_identifier.py
```

## Features

- Image preprocessing
- Data loading and transformation
- Custom CNN architecture
- Performance evaluation (F1-score, accuracy)
- Result visualization

## Metrics

The project includes various performance metrics:
- Accuracy
- F1-score
- Confusion Matrix
- ROC Curve

## Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

