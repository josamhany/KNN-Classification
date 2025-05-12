# KNN Classification on Wine Dataset

## Overview
This project implements a K-Nearest Neighbors (KNN) classification model on the Wine dataset from scikit-learn. The dataset consists of 178 samples, 13 features, and 3 classes representing different wine cultivars. The goal is to classify wines into their respective categories using KNN, with additional steps for data preprocessing, feature selection, and model evaluation.
 
## Dataset
The Wine dataset from scikit-learn is used in this project. It includes:
- **Number of Samples**: 178
- **Number of Features**: 13 (e.g., alcohol, malic acid, ash, etc.)
- **Number of Classes**: 3 (class_0, class_1, class_2)

## Project Structure
- `KNN_Classification.ipynb`: Jupyter Notebook containing the implementation of the KNN classification pipeline, including data loading, exploratory data analysis (EDA), preprocessing, model training, and evaluation.
- `README.md`: This file, providing an overview of the project.

## Requirements
To run this project, you need the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `micropip` (for Pyodide environment)

You can install the dependencies using:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn micropip
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/knn-classification-wine.git
   cd knn-classification-wine
   ```
2. Open the `KNN_Classification.ipynb` notebook in Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook KNN_Classification.ipynb
   ```
3. Run the cells sequentially to load the dataset, perform EDA, preprocess the data, train the KNN model, and evaluate its performance.

## Methodology
1. **Data Loading**: The Wine dataset is loaded using `sklearn.datasets.load_wine()`.
2. **Exploratory Data Analysis (EDA)**:
   - Feature correlation heatmap to identify relationships between features.
   - Feature distribution histograms to understand the data distribution (as shown in the provided figure).
3. **Preprocessing**:
   - StandardScaler for feature scaling.
   - VarianceThreshold for feature selection.
   - PCA for dimensionality reduction (if applicable).
4. **Model Training**:
   - KNN classifier is implemented using `sklearn.neighbors.KNeighborsClassifier`.
   - The dataset is split into training and testing sets using `train_test_split`.
5. **Evaluation**:
   - Cross-validation scores to assess model performance.
   - Metrics: accuracy, precision, recall, F1-score, and confusion matrix.

## Results
- The feature distribution histograms provide insights into the dataset's characteristics (refer to the provided figure in the notebook).
- Detailed model performance metrics will be displayed upon running the notebook, including accuracy, precision, recall, F1-score, and a confusion matrix.

## License
This project is licensed under the MIT License. See the [LinkedIn](https://www.linkedin.com/posts/josam-hany-76b449301_machinelearning-datascience-knn-activity-7327682512460095488-5oyu?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE0hRQMBJwwXzE_2WIlbIlC2-W8nTypJdkU) file for details.

## Contact
For any questions or contributions, please contact the team members via GitHub or email.
