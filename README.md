# Sentiment Analysis

This repository is dedicated to storing and processing a dataset of reviews of tutors and their corresponding sentiment classes. 
The data will be used for fine-tuning the model `tabularisai/multilingual-sentiment-analysis` using transfer learning.

## Project Structure

- `data/`: Contains the dataset (Excel file) with reviews and sentiment labels.
- `notebooks/`: Collab notebooks for data exploration and model fine-tuning.
- `src/`: Python source code files for data preprocessing and model training.
- `requirements.txt`: List of dependencies for the project.

## Dataset

The dataset consists of reviews of tutors with associated sentiment labels. The reviews are in Polish language, and the sentiment labels can be one of the following: `Very Negative`, `Negative`, `Neutral`, `Positive`, `Very Positive`.

The dataset is stored in an Excel file located in the `data/` folder.

## How to Use

### Setting Up

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis.git
   cd sentiment-analysis

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\Activate.bat

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt

### Notebooks
Collab notebooks available in the notebooks/ folder to explore the data and train the model.

### Model
To fine-tune the tabularisai/multilingual-sentiment-analysis model, use the script located in the src/ folder.
