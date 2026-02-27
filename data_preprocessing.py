"""
Data Preprocessing Module for Diabetes Prediction Project
NIH-K43 Community Screening Dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Handles data loading, cleaning, and preprocessing for diabetes prediction.
    """
    
    def __init__(self, file_path):
        """
        Initialize the preprocessor with the dataset path.
        
        Args:
            file_path (str): Path to the dataset file (CSV, Excel, etc.)
        """
        self.file_path = file_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.label_encoders = {}
        
    def load_data(self):
        """
        Load dataset from file.
        Supports CSV, Excel formats.
        """
        try:
            if self.file_path.endswith('.csv'):
                self.data = pd.read_csv(self.file_path)
            elif self.file_path.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(self.file_path)
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel.")
            
            print(f"✓ Dataset loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            return self.data
        except FileNotFoundError:
            print(f"✗ Error: File not found at {self.file_path}")
            print("Please ensure the dataset is placed in the 'data' folder.")
            return None
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            return None
    
    def explore_data(self):
        """
        Display basic information about the dataset.
        """
        if self.data is None:
            print("✗ No data loaded. Please load data first.")
            return
        
        print("\n" + "="*70)
        print("DATASET OVERVIEW")
        print("="*70)
        print(f"\nShape: {self.data.shape}")
        print(f"\nColumns: {list(self.data.columns)}")
        print("\nFirst few rows:")
        print(self.data.head())
        print("\nData types:")
        print(self.data.dtypes)
        print("\nMissing values:")
        print(self.data.isnull().sum())
        print("\nBasic statistics:")
        print(self.data.describe())
        
    def handle_missing_values(self, strategy='mean'):
        """
        Handle missing values in the dataset.
        
        Args:
            strategy (str): Strategy for imputation ('mean', 'median', 'most_frequent')
        """
        if self.data is None:
            print("✗ No data loaded.")
            return
        
        missing_before = self.data.isnull().sum().sum()
        
        # Separate numerical and categorical columns
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        # Impute numerical columns
        if len(numerical_cols) > 0:
            num_imputer = SimpleImputer(strategy=strategy)
            self.data[numerical_cols] = num_imputer.fit_transform(self.data[numerical_cols])
        
        # Impute categorical columns
        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            self.data[categorical_cols] = cat_imputer.fit_transform(self.data[categorical_cols])
        
        missing_after = self.data.isnull().sum().sum()
        print(f"✓ Missing values handled: {missing_before} → {missing_after}")
    
    def encode_categorical_variables(self, target_column):
        """
        Encode categorical variables using Label Encoding.
        
        Args:
            target_column (str): Name of the target column
        """
        if self.data is None:
            print("✗ No data loaded.")
            return
        
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != target_column]
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col].astype(str))
            self.label_encoders[col] = le
        
        print(f"✓ Encoded {len(categorical_cols)} categorical variables")
    
    def prepare_features_and_target(self, target_column, exclude_columns=None):
        """
        Separate features (X) and target (y).
        
        Args:
            target_column (str): Name of the target column
            exclude_columns (list): List of columns to exclude from features
        """
        if self.data is None:
            print("✗ No data loaded.")
            return
        
        # Handle target encoding if it's categorical
        if self.data[target_column].dtype == 'object':
            le = LabelEncoder()
            self.data[target_column] = le.fit_transform(self.data[target_column])
            print(f"✓ Target variable encoded: {dict(enumerate(le.classes_))}")
        
        # Separate features and target
        exclude_cols = [target_column]
        if exclude_columns:
            exclude_cols.extend(exclude_columns)
        
        X = self.data.drop(columns=exclude_cols)
        y = self.data[target_column]
        
        self.feature_names = X.columns.tolist()
        print(f"✓ Features prepared: {len(self.feature_names)} features")
        print(f"  Features: {self.feature_names}")
        print(f"✓ Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size (float): Proportion of test set
            random_state (int): Random seed for reproducibility
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"✓ Data split completed:")
        print(f"  Training set: {self.X_train.shape[0]} samples")
        print(f"  Testing set: {self.X_test.shape[0]} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self):
        """
        Standardize features using StandardScaler.
        """
        if self.X_train is None or self.X_test is None:
            print("✗ Data not split yet. Please split data first.")
            return
        
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print("✓ Features scaled using StandardScaler")
        
        return self.X_train, self.X_test
    
    def preprocess_pipeline(self, target_column, exclude_columns=None, 
                           test_size=0.2, scale=True, missing_strategy='mean'):
        """
        Complete preprocessing pipeline.
        
        Args:
            target_column (str): Name of the target column
            exclude_columns (list): Columns to exclude from features
            test_size (float): Proportion of test set
            scale (bool): Whether to scale features
            missing_strategy (str): Strategy for handling missing values
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_names)
        """
        print("\n" + "="*70)
        print("STARTING PREPROCESSING PIPELINE")
        print("="*70 + "\n")
        
        # Load data if not already loaded
        if self.data is None:
            self.load_data()
            if self.data is None:
                return None
        
        # Handle missing values
        self.handle_missing_values(strategy=missing_strategy)
        
        # Encode categorical variables
        self.encode_categorical_variables(target_column)
        
        # Prepare features and target
        X, y = self.prepare_features_and_target(target_column, exclude_columns)
        
        # Split data
        self.split_data(X, y, test_size=test_size)
        
        # Scale features
        if scale:
            self.scale_features()
        
        print("\n" + "="*70)
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("="*70 + "\n")
        
        return self.X_train, self.X_test, self.y_train, self.y_test, self.feature_names


# Example usage
if __name__ == "__main__":
    # Example: Adjust the file path and column names based on your dataset
    preprocessor = DataPreprocessor('data/diabetes_dataset.csv')
    
    # Option 1: Run complete pipeline
    result = preprocessor.preprocess_pipeline(
        target_column='diabetes_status',  # Adjust to your target column name
        exclude_columns=None,  # Add any ID columns to exclude
        test_size=0.2,
        scale=True
    )
    
    if result:
        X_train, X_test, y_train, y_test, feature_names = result
        print(f"\nReady for modeling!")
        print(f"Training set shape: {X_train.shape}")
        print(f"Testing set shape: {X_test.shape}")
