# Drug Side-Effect Rating Classification from Patient Reviews

This project trains traditional machine learning models to classify patient-reported side-effect ratings into five categories based on review text.

## Project Structure
```text
textdrug-side-effect-classifier/
├── data/
│   ├── drugLibTrain_raw.tsv
│   └── drugLibTest_raw.tsv
├── train_model.py
├── app.py
├── side_effect_model.joblib (generated after running training)
└── requirements.txt
```

## Instructions
1. Download `drugLibTrain_raw.tsv` and `drugLibTest_raw.tsv` from the UCI Machine Learning Repository (Drug Reviews dataset) and place them inside the `data/` folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python train_model.py
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
