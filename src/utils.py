"""
Utility functions for car insurance claim prediction
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def remove_outliers_iqr(data, column):
    """
    Remove outliers from a column using the IQR method.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset
    column : str
        Column name to remove outliers from
        
    Returns:
    --------
    pandas.DataFrame
        Data without outliers
    int
        Number of outliers removed
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Count outliers
    outliers = ((data[column] < lower_bound) | (data[column] > upper_bound)).sum()
    
    # Filter data
    data_clean = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return data_clean, outliers


def balance_dataset(data, target_column, method='undersample', random_state=42):
    """
    Balance dataset by target variable.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset
    target_column : str
        Name of target column
    method : str
        'undersample' or 'oversample'
    random_state : int
        Random seed
        
    Returns:
    --------
    pandas.DataFrame
        Balanced dataset
    """
    from sklearn.utils import resample
    
    # Separate classes
    df_class_0 = data[data[target_column] == 0]
    df_class_1 = data[data[target_column] == 1]
    
    if method == 'undersample':
        # Downsample majority class
        if len(df_class_0) > len(df_class_1):
            df_class_0_downsampled = resample(df_class_0,
                                              replace=False,
                                              n_samples=len(df_class_1),
                                              random_state=random_state)
            df_balanced = pd.concat([df_class_0_downsampled, df_class_1])
        else:
            df_class_1_downsampled = resample(df_class_1,
                                              replace=False,
                                              n_samples=len(df_class_0),
                                              random_state=random_state)
            df_balanced = pd.concat([df_class_0, df_class_1_downsampled])
    else:
        raise ValueError("Only 'undersample' method is currently supported")
    
    return df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)


def evaluate_model(y_true, y_pred, model_name='Model'):
    """
    Evaluate a classification model.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\\n=== {model_name} Evaluation ===")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Claim', 'Claim']))
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', cmap='Blues'):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    title : str
        Plot title
    cmap : str
        Colormap
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=['No Claim', 'Claim'],
                yticklabels=['No Claim', 'Claim'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_names, importances, top_n=10, title='Feature Importance'):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    feature_names : list
        List of feature names
    importances : array-like
        Feature importance scores
    top_n : int
        Number of top features to display
    title : str
        Plot title
    """
    # Create dataframe
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(feature_df['Feature'], feature_df['Importance'], color='coral', alpha=0.7)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def compare_models(model_results):
    """
    Compare multiple models.
    
    Parameters:
    -----------
    model_results : dict
        Dictionary with model names as keys and accuracy as values
        Example: {'Decision Tree': 0.78, 'SVM': 0.80}
    """
    models = list(model_results.keys())
    accuracies = list(model_results.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'coral'][:len(models)], 
                   alpha=0.7, edgecolor='black')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.ylim([min(accuracies) - 0.1, 1.0])
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{acc:.4f}\\n({acc*100:.2f}%)',
                 ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
