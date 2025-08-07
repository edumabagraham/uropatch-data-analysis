import numpy as np
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from collections import defaultdict
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class NestedCVOptimizer:
    def __init__(self, X, y, groups, n_outer_folds=5, n_inner_folds=3, 
                n_trials=100, random_state=42):
        """
        Initialize the nested cross-validation optimizer.
        
        Parameters:
        - X: Feature matrix
        - y: Target labels
        - groups: Group labels for GroupKFold
        - n_outer_folds: Number of outer CV folds
        - n_inner_folds: Number of inner CV folds
        - n_trials: Number of Optuna trials per inner fold
        - random_state: Random state for reproducibility
        """
        self.X = X
        self.y = y
        self.groups = groups
        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.n_trials = n_trials
        self.random_state = random_state

            
        # Initialize CV splitters
        self.outer_cv = GroupKFold(n_splits=n_outer_folds)
        self.inner_cv = GroupKFold(n_splits=n_inner_folds)
        
        # Store results
        self.outer_scores = []
        self.best_configs = []
        
    def define_search_space(self, trial):
        """
        Define the hyperparameter search space for Optuna.
        """
        # Model selection
        model_name = trial.suggest_categorical('model', ['rf', 'xgb'])
        
        # Imbalance handling technique
        imbalance_technique = trial.suggest_categorical(
            'imbalance_technique', ['none','smote', 'smote_tomek', 'tomek_links']
        )
        
        # Model-specific hyperparameters
        if model_name == 'rf':
            params = {
                'model': 'rf',
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', 
                                                        ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'imbalance_technique': imbalance_technique
            }
        else:  # XGBoost
            params = {
                'model': 'xgb',
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'imbalance_technique': imbalance_technique
            }
                
        return params
    
    def apply_imbalance_handling(self, X_train, y_train, technique):
        """
        Apply the specified imbalance handling technique.
        """
        if technique == 'none':
            return X_train, y_train
        elif technique == 'smote':
            sampler = SMOTE(random_state=self.random_state)
            return sampler.fit_resample(X_train, y_train)
        elif technique == 'smote_tomek':
            sampler = SMOTETomek(random_state=self.random_state)
            return sampler.fit_resample(X_train, y_train)
        elif technique == 'tomek_links':
            sampler = TomekLinks()
            return sampler.fit_resample(X_train, y_train)
        else:
            raise ValueError(f"Unknown imbalance technique: {technique}")
    
    def create_model(self, params):
        """
        Create a model based on the parameters.
        """
        if params['model'] == 'rf':
            return RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=params['max_features'],
                bootstrap=params['bootstrap'],
                random_state=self.random_state,
                n_jobs=-1
            )
        elif params['model'] == 'xgb':
            return XGBClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                min_child_weight=params['min_child_weight'],
                gamma=params['gamma'],
                reg_alpha=params['reg_alpha'],
                reg_lambda=params['reg_lambda'],
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss',  # Suppress warnings
                verbosity=0  # Suppress XGBoost output
            )
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate comprehensive metrics for model evaluation.
        """

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', pos_label='void', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', pos_label='void', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='binary', pos_label='void', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        

                
        return metrics
    
    def inner_cv_objective(self, trial, X_train_outer, y_train_outer, groups_train_outer):
        """
        Objective function for inner cross-validation (hyperparameter optimization).
        """
        params = self.define_search_space(trial)
        
        fold_scores = []
        
        for inner_train_idx, inner_val_idx in self.inner_cv.split(
            X_train_outer, y_train_outer, groups_train_outer
        ):
            
            # Split data
            X_inner_train = X_train_outer.iloc[inner_train_idx]
            X_inner_val = X_train_outer.iloc[inner_val_idx]
            y_inner_train = y_train_outer.iloc[inner_train_idx]
            y_inner_val = y_train_outer.iloc[inner_val_idx]
            
            
            
            # Apply imbalance handling
            X_inner_train_balanced, y_inner_train_balanced = self.apply_imbalance_handling(
                X_inner_train, y_inner_train, params['imbalance_technique']
            )
            
            # Encode labels
            label_encoder = LabelEncoder()
            y_inner_train_encoded = label_encoder.fit_transform(y_inner_train_balanced)
            
            # Create and train model
            model = self.create_model(params)
            # model.fit(X_inner_train_balanced, y_inner_train_balanced)
            model.fit(X_inner_train_balanced, y_inner_train_encoded)
            
            
            # Predict and calculate score
            # y_pred = model.predict(X_inner_val)
            y_pred = label_encoder.inverse_transform(model.predict(X_inner_val))
            
            # Use F1 score of positive class as the optimization metric (binary classification)
            score = f1_score(y_inner_val, y_pred, pos_label='void', zero_division=0)
            fold_scores.append(score)
        
        return np.mean(fold_scores)
    
    def run_nested_cv(self):
        """
        Run the nested cross-validation procedure.
        """
        print("Starting Nested Cross-Validation...")
        print(f"Outer folds: {self.n_outer_folds}, Inner folds: {self.n_inner_folds}")
        print(f"Total unique groups: {len(np.unique(self.groups))}")
        
        for fold_idx, (train_idx, test_idx) in enumerate(self.outer_cv.split(self.X, self.y, self.groups)):
            print(f"\n=== Outer Fold {fold_idx + 1}/{self.n_outer_folds} ===")
            
            # Split data for outer fold
            X_train_outer = self.X.iloc[train_idx]
            X_test_outer = self.X.iloc[test_idx]
            y_train_outer = self.y.iloc[train_idx]
            y_test_outer = self.y.iloc[test_idx]
            groups_train_outer = self.groups.iloc[train_idx]
            
            print(f"Training samples: {len(X_train_outer)}, Test samples: {len(X_test_outer)}")
            print(f"Training groups: {len(np.unique(groups_train_outer))}")
            
            # Inner cross-validation for hyperparameter optimization
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
            )
            
            print(f"Running hyperparameter optimization with {self.n_trials} trials...")
            study.optimize(
                lambda trial: self.inner_cv_objective(trial, X_train_outer, y_train_outer, groups_train_outer),
                n_trials=self.n_trials,
                show_progress_bar=True
            )
            
            best_params = study.best_params
            print(f"Best parameters: {best_params}")
            print(f"Best inner CV score: {study.best_value:.4f}")
            

            
            # Apply imbalance handling
            X_train_balanced, y_train_balanced = self.apply_imbalance_handling(
                X_train_outer, y_train_outer, best_params['imbalance_technique']
            )
            
            # Encode labels
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train_balanced)
            
            # Create and train final model
            final_model = self.create_model(best_params)
            final_model.fit(X_train_balanced, y_train_encoded)
            
            y_pred = label_encoder.inverse_transform(final_model.predict(X_test_outer))
            try:
                y_pred_proba = final_model.predict_proba(X_test_outer)
            except AttributeError:
                y_pred_proba = None
            
            # Calculate metrics
            fold_metrics = self.calculate_metrics(y_test_outer, y_pred, y_pred_proba)
            
            print(f"Outer fold test metrics:")
            for metric, value in fold_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            # Store results
            self.outer_scores.append(fold_metrics)
            self.best_configs.append(best_params)
        
        return self._summarize_results()
    
    def _summarize_results(self):
        """
        Summarize the nested CV results.
        """
        print("\n" + "="*50)
        print("NESTED CROSS-VALIDATION RESULTS SUMMARY")
        print("="*50)
        
        # Convert results to DataFrame for easier analysis
        results_df = pd.DataFrame(self.outer_scores)
        
        print("\nOverall Performance (Mean ± Std across outer folds):")
        for metric in results_df.columns:
            mean_score = results_df[metric].mean()
            std_score = results_df[metric].std()
            print(f"  {metric}: {mean_score:.4f} ± {std_score:.4f}")
        
        # Additional reporting for binary classification
        print(f"\nOptimization Metric (F1 of Positive Class):")
        if 'f1' in results_df.columns:
            # Also show how the positive class F1 compares to macro F1
            print(f"  Mean F1: {results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}")
            print("  (Note: Optimization used F1 of positive class, not macro F1)")
        
        # Analyze best configurations
        print("\nBest Configurations per Fold:")
        config_summary = defaultdict(list)
        for i, config in enumerate(self.best_configs):
            print(f"  Fold {i+1}: Model={config['model']}, "
                f"Imbalance={config['imbalance_technique']}")
            config_summary['model'].append(config['model'])
            config_summary['imbalance_technique'].append(config['imbalance_technique'])
        
        print("\nConfiguration Frequency:")
        for key, values in config_summary.items():
            unique_vals, counts = np.unique(values, return_counts=True)
            print(f"  {key}:")
            for val, count in zip(unique_vals, counts):
                print(f"    {val}: {count}/{len(values)} folds")
        
        return {
            'mean_scores': results_df.mean().to_dict(),
            'std_scores': results_df.std().to_dict(),
            'individual_scores': self.outer_scores,
            'best_configs': self.best_configs,
            'summary_df': results_df
        }


    

    
