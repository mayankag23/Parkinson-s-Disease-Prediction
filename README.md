# Parkinson's Disease Predictor

This project uses machine learning techniques to predict the presence of Parkinson's disease based on voice and audio recording features. The analysis is implemented in a Jupyter Notebook and utilizes a publicly available dataset of biomedical voice measurements from patients with and without Parkinson’s disease.

## Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- **Contents:** 195 instances, 24 columns (features from voice recordings, including frequency, jitter, shimmer, noise-to-harmonics ratio, etc.)
- **Label:** `status` (1 = Parkinson's, 0 = Healthy)

## Workflow

1. **Data Loading and Exploration**
    - The dataset is loaded using `pandas`.
    - Preliminary exploration is performed to understand the shape, check for missing values, and summarize statistics of the features.

2. **Preprocessing**
    - The dataset is confirmed to have no missing values.
    - All features except the `name` column are used for modeling.

3. **Feature Selection and Splitting**
    - Features (`X`) are selected from the biomedical measurements.
    - The target (`y`) is the `status` column.
    - Data is split into training and test sets using `train_test_split`.

4. **Modeling**
    - Machine learning models such as **Logistic Regression** and **Random Forest Classifier** are trained on the data.
    - Models are evaluated using accuracy score and confusion matrix.

5. **Visualization**
    - Data visualization is conducted using `matplotlib` and `seaborn` to analyze feature distributions and model performance.

## Technologies Used

- Python 3
- Jupyter Notebook
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/NMalik31/Parkinson-s-Disease-Predictor.git
    ```
2. Open `Project_53_Parkinson's_Disease_Predictor.ipynb` in Jupyter Notebook or Google Colab.
3. Ensure the dataset (`parkinsons.data`) is available at the specified path or adjust the notebook path accordingly.
4. Run the notebook cells to reproduce the analysis and predictions.

## File Structure

- `Project_53_Parkinson's_Disease_Predictor.ipynb` — Main analysis notebook.
- `parkinsons.data` — Voice measurement dataset (not included; download from [UCI Repository](https://archive.ics.uci.edu/ml/datasets/parkinsons)).

## Results

- The notebook demonstrates how machine learning models can effectively classify whether a patient has Parkinson's disease based on voice features, achieving high accuracy on the provided dataset.

## References

- [UCI Parkinson's Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- [Notebook on GitHub](https://github.com/NMalik31/Parkinson-s-Disease-Predictor/blob/main/Project_53_Parkinson's_Disease_Predictor.ipynb)
