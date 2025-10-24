"""Sklearn pipelines for telco churn prediction."""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier


# Feature definitions
NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]


def get_preprocessor():
    """Create preprocessing pipeline."""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    
    return preprocessor


def get_pipeline(model_type: str = 'hgb'):
    """
    Create full pipeline with preprocessing + model.
    
    Args:
        model_type: 'dummy', 'logreg', or 'hgb'
    
    Returns:
        sklearn Pipeline
    """
    preprocessor = get_preprocessor()
    
    if model_type == 'dummy':
        model = DummyClassifier(strategy='stratified', random_state=42)
    elif model_type == 'logreg':
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
    elif model_type == 'hgb':
        model = HistGradientBoostingClassifier(
            class_weight='balanced',
            max_iter=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    return pipeline