"""
Task 4: Most Complex Python Code - Machine Learning Pipeline
Advanced implementation of a complete ML pipeline with feature engineering,
model training, hyperparameter tuning, cross-validation, and evaluation.

This code demonstrates:
- Advanced OOP principles
- Data preprocessing and feature engineering
- Multiple ML algorithms with comparison
- Cross-validation and hyperparameter tuning
- Performance metrics and evaluation
- Model persistence
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any
from abc import ABC, abstractmethod
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, cross_validate
)
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, PolynomialFeatures,
    LabelEncoder
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc, StratifiedKFold as SkFold
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Advanced feature engineering transformer combining multiple
    feature extraction and transformation techniques.
    """
    
    def __init__(self, poly_degree: int = 2, interaction_features: bool = True):
        self.poly_degree = poly_degree
        self.interaction_features = interaction_features
        self.poly_transformer = None
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        self.poly_transformer = PolynomialFeatures(
            degree=self.poly_degree,
            include_bias=False
        )
        self.poly_transformer.fit(X)
        return self
    
    def transform(self, X):
        X_poly = self.poly_transformer.transform(X)
        
        # Add custom engineered features
        if self.interaction_features and X.shape[1] >= 2:
            X_interactions = []
            n_features = X.shape[1]
            for i in range(n_features):
                for j in range(i+1, n_features):
                    X_interactions.append(X[:, i] * X[:, j])
            
            if X_interactions:
                X_interactions = np.column_stack(X_interactions)
                X_poly = np.hstack([X_poly, X_interactions])
        
        return X_poly


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Intelligent feature selection using multiple techniques."""
    
    def __init__(self, n_features: int = 10, method: str = 'kbest'):
        self.n_features = n_features
        self.method = method
        self.selector = None
    
    def fit(self, X, y):
        if self.method == 'kbest':
            self.selector = SelectKBest(f_classif, k=min(self.n_features, X.shape[1]))
        elif self.method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            self.selector = RFE(estimator, n_features_to_select=min(self.n_features, X.shape[1]))
        
        self.selector.fit(X, y)
        return self
    
    def transform(self, X):
        return self.selector.transform(X)


class MLModel:
    """Advanced ML model wrapper with comprehensive evaluation and tuning."""
    
    def __init__(self, model_name: str = 'random_forest'):
        self.model_name = model_name
        self.model = None
        self.pipeline = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.metrics = {}
        self.best_params = None
    
    def _create_base_model(self):
        """Create base model based on model_name."""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='lbfgs'
            )
        }
        
        return models.get(self.model_name, models['random_forest'])
    
    def create_pipeline(self, use_feature_engineering: bool = True):
        """Create a complete ML pipeline."""
        steps = [
            ('scaler', StandardScaler()),
        ]
        
        if use_feature_engineering:
            steps.append(('feature_engineer', AdvancedFeatureEngineer(poly_degree=2)))
            steps.append(('feature_selector', FeatureSelector(n_features=15, method='kbest')))
        
        steps.append(('model', self._create_base_model()))
        
        self.pipeline = Pipeline(steps)
        return self.pipeline
    
    def fit(self, X_train, y_train, cv_folds: int = 5):
        """Train the model with cross-validation."""
        if self.pipeline is None:
            self.create_pipeline()
        
        print(f"\n{'='*60}")
        print(f"Training {self.model_name} with Cross-Validation (CV={cv_folds})")
        print(f"{'='*60}")
        
        # Cross-validation score
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }
        
        cv_scores = cross_validate(
            self.pipeline, X_train, y_train,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring=scoring,
            return_train_score=True
        )
        
        # Print CV results
        print("\nCross-Validation Results:")
        print("-" * 60)
        for metric in scoring.keys():
            train_score = cv_scores[f'train_{metric}'].mean()
            test_score = cv_scores[f'test_{metric}'].mean()
            std = cv_scores[f'test_{metric}'].std()
            print(f"{metric:15} | Train: {train_score:.4f} | Test: {test_score:.4f} ± {std:.4f}")
        
        # Train on full training set
        self.pipeline.fit(X_train, y_train)
        print("\n✓ Model training completed.")
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform grid search for hyperparameter tuning."""
        print(f"\nHyperparameter Tuning for {self.model_name}...")
        print("-" * 60)
        
        param_grids = {
            'random_forest': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [5, 10, 15],
                'model__min_samples_split': [2, 5, 10],
            },
            'gradient_boosting': {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.5],
                'model__max_depth': [3, 5, 7],
            },
            'svm': {
                'model__C': [0.1, 1, 10],
                'model__gamma': ['scale', 'auto'],
                'model__kernel': ['rbf', 'poly'],
            },
            'logistic_regression': {
                'model__C': [0.1, 1, 10],
                'model__solver': ['lbfgs', 'liblinear'],
            }
        }
        
        param_grid = param_grids.get(self.model_name, {})
        
        if not param_grid:
            print("No hyperparameters to tune for this model.")
            return
        
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        self.pipeline = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
    
    def evaluate(self, X_test, y_test):
        """Comprehensive model evaluation."""
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1] if hasattr(self.pipeline, 'predict_proba') else None
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
        }
        
        if y_pred_proba is not None:
            try:
                self.metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            except:
                self.metrics['roc_auc'] = None
        
        print(f"\n{'='*60}")
        print(f"Evaluation Metrics for {self.model_name}")
        print(f"{'='*60}")
        for metric, value in self.metrics.items():
            if value is not None:
                print(f"{metric:15}: {value:.4f}")
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    def predict(self, X):
        """Make predictions."""
        return self.pipeline.predict(X)


class MLPipelineComparison:
    """Compare multiple ML models and algorithms."""
    
    def __init__(self, dataset: str = 'iris'):
        self.dataset_name = dataset
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
    
    def load_dataset(self):
        """Load dataset."""
        if self.dataset_name == 'iris':
            data = load_iris()
        elif self.dataset_name == 'breast_cancer':
            data = load_breast_cancer()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        self.X = data.data
        self.y = data.target
        
        print(f"✓ Loaded {self.dataset_name} dataset")
        print(f"  Shape: {self.X.shape}")
        print(f"  Classes: {len(np.unique(self.y))}")
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        """Split data into train and test sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y
        )
        
        print(f"✓ Data split: Train={self.X_train.shape[0]}, Test={self.X_test.shape[0]}")
    
    def train_models(self, model_names: List[str] = None):
        """Train multiple models."""
        if model_names is None:
            model_names = ['random_forest', 'gradient_boosting', 'svm', 'logistic_regression']
        
        for model_name in model_names:
            print(f"\n{'#'*60}")
            print(f"# Training {model_name.upper()}")
            print(f"{'#'*60}")
            
            model = MLModel(model_name)
            model.create_pipeline(use_feature_engineering=True)
            model.fit(self.X_train, self.y_train, cv_folds=5)
            model.evaluate(self.X_test, self.y_test)
            
            self.models[model_name] = model
            self.results[model_name] = model.metrics
    
    def compare_models(self):
        """Compare all trained models."""
        print(f"\n{'='*80}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.sort_values('f1', ascending=False)
        
        print(comparison_df.to_string())
        
        best_model = comparison_df['f1'].idxmax()
        print(f"\n✓ Best model: {best_model} (F1-Score: {comparison_df.loc[best_model, 'f1']:.4f})")
        
        return comparison_df


def main():
    """Main function demonstrating the complex ML pipeline."""
    print("\n" + "="*60)
    print("TASK 4: MOST COMPLEX PYTHON CODE")
    print("Advanced Machine Learning Pipeline with Hyperparameter Tuning")
    print("="*60)
    
    # Initialize and run the pipeline
    pipeline = MLPipelineComparison(dataset='iris')
    pipeline.load_dataset()
    pipeline.split_data()
    pipeline.train_models(
        model_names=['random_forest', 'gradient_boosting', 'svm', 'logistic_regression']
    )
    comparison_df = pipeline.compare_models()
    
    print("\n✓ Advanced ML Pipeline completed successfully!")


if __name__ == "__main__":
    main()
