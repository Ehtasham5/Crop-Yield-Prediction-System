# Smart Crop Yield Prediction System

This project follows the assignment instructions:

1. Topic selected and dataset downloaded from Kaggle.
2. Dataset cleaned and preprocessed.
3. Model trained and tested.
4. Trained model saved.
5. Simple Streamlit frontend connected to the model.

## Dataset

Kaggle dataset: **Smart Crop yield predication dataset** by Arif Miah  
Kaggle slug: `miadul/smart-crop-yield-predication-dataset`

The local dataset file is:

```text
data/raw/crop_yield_dataset.csv
```

The model predicts `Yield_ton_per_ha` using crop type, region, soil type, soil pH, rainfall, temperature, humidity, fertilizer usage, irrigation type, pesticide usage, planting density, and previous crop.

## Project Structure

```text
app.py                              Streamlit frontend
src/download_data.py                Kaggle dataset downloader
src/train_model.py                  Cleaning, preprocessing, training, testing
data/raw/crop_yield_dataset.csv     Kaggle dataset
data/processed/cleaned_crop_yield_data.csv
models/crop_yield_model.joblib      Saved trained model
models/model_metrics.json           Test metrics
notebooks/crop_yield_training.ipynb Notebook training workflow
requirements.txt
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset from Kaggle:

```bash
python src/download_data.py
```

Train and save the model:

```bash
python src/train_model.py
```

Run the Streamlit app:

```bash
streamlit run app.py
```
