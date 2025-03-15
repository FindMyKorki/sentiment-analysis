import pandas as pd
from transformers import pipeline

pipe = pipeline("sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis")

file_path = "./data/Data.xlsx"
df = pd.read_excel(file_path)

def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

def preprocess_data(df):
    X = df['Opinia'].values
    y = df['Klasa'].values
    return X, y

def fine_tune_model(X_train, y_train):
    # to do

if __name__ == "__main__":
    data = load_data(data_file_path)
    
    X_train, y_train = preprocess_data(data)
    
    #fine_tune_model(X_train, y_train)
