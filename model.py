"""
Enhanced NFL Prediction Model using XGBoost with hyperparameter tuning
"""
import numpy as np
import joblib
import os
from typing import Tuple, Dict, Any, Optional
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, mean_absolute_error

class NFLPredictionModel:
    """XGBoost-based NFL prediction model with betting line features"""
    
    def __init__(self):
        self.win_model: Optional[XGBClassifier] = None
        self.spread_model: Optional[XGBClassifier] = None
        self.total_model: Optional[XGBRegressor] = None
        self.feature_importance: Dict[str, float] = {}
        self.is_trained: bool = False
        self.training_accuracy: float = 0.0
        self.cv_accuracy: float = 0.0
        
    def train(
        self, 
        X: np.ndarray, 
        y_win: np.ndarray, 
        y_spread: Optional[np.ndarray] = None,
        y_total: Optional[np.ndarray] = None,
        tune_hyperparameters: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model with optional hyperparameter tuning.
        
        Args:
            X: Feature matrix
            y_win: Binary win/loss labels (1 = home win)
            y_spread: Binary spread cover labels (optional)
            y_total: Continuous total points (optional)
            tune_hyperparameters: Whether to run grid search
        """
        results = {}
        
        # Define base parameters
        base_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'random_state': 42,
            'n_jobs': -1,
        }
        
        if tune_hyperparameters and len(X) >= 100:
            print("Running hyperparameter tuning...")
            
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9, 1.0],
            }
            
            # Simplified grid for faster training
            simple_grid = {
                'n_estimators': [100, 200],
                'max_depth': [4, 6],
                'learning_rate': [0.05, 0.1],
            }
            
            grid_search = GridSearchCV(
                XGBClassifier(**base_params),
                simple_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X, y_win)
            
            best_params = grid_search.best_params_
            print(f"Best parameters: {best_params}")
            results['best_params'] = best_params
            
            self.win_model = grid_search.best_estimator_
        else:
            # Use default good parameters
            default_params = {
                **base_params,
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'min_child_weight': 3,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
            }
            self.win_model = XGBClassifier(**default_params)
            self.win_model.fit(X, y_win)
        
        # Calculate training accuracy
        train_preds = self.win_model.predict(X)
        self.training_accuracy = accuracy_score(y_win, train_preds)
        results['training_accuracy'] = self.training_accuracy
        
        # Cross-validation accuracy
        cv_scores = cross_val_score(self.win_model, X, y_win, cv=5, scoring='accuracy')
        self.cv_accuracy = cv_scores.mean()
        results['cv_accuracy'] = self.cv_accuracy
        results['cv_std'] = cv_scores.std()
        
        print(f"Training Accuracy: {self.training_accuracy:.4f}")
        print(f"CV Accuracy: {self.cv_accuracy:.4f} (+/- {cv_scores.std():.4f})")
        
        # Train spread model if labels provided
        if y_spread is not None:
            spread_params = {
                **base_params,
                'n_estimators': 150,
                'max_depth': 4,
                'learning_rate': 0.05,
            }
            self.spread_model = XGBClassifier(**spread_params)
            self.spread_model.fit(X, y_spread)
            spread_acc = accuracy_score(y_spread, self.spread_model.predict(X))
            results['spread_accuracy'] = spread_acc
            print(f"Spread Model Accuracy: {spread_acc:.4f}")
        
        # Train total model if labels provided
        if y_total is not None:
            total_params = {
                'objective': 'reg:squarederror',
                'n_estimators': 150,
                'max_depth': 4,
                'learning_rate': 0.05,
                'random_state': 42,
            }
            self.total_model = XGBRegressor(**total_params)
            self.total_model.fit(X, y_total)
            total_mae = mean_absolute_error(y_total, self.total_model.predict(X))
            results['total_mae'] = total_mae
            print(f"Total Model MAE: {total_mae:.2f} points")
        
        # Extract feature importance
        self._extract_feature_importance()
        results['feature_importance'] = self.feature_importance
        
        self.is_trained = True
        return results
    
    def _extract_feature_importance(self):
        """Extract and store feature importance from win model"""
        if self.win_model is None:
            return
            
        from features import FEATURE_NAMES
        
        importances = self.win_model.feature_importances_
        self.feature_importance = {}
        
        for i, name in enumerate(FEATURE_NAMES):
            if i < len(importances):
                self.feature_importance[name] = float(importances[i])
        
        # Sort by importance
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        print("\nTop 10 Feature Importances:")
        for i, (name, imp) in enumerate(self.feature_importance.items()):
            if i >= 10:
                break
            print(f"  {name}: {imp:.4f}")
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Make predictions for a single game or batch.
        
        Returns dict with win_prob, spread_prob, predicted_total, confidence, recommendation
        """
        if not self.is_trained or self.win_model is None:
            return self._heuristic_predict(X)
        
        # Ensure 2D array
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Win probability
        win_probs = self.win_model.predict_proba(X)[:, 1]
        
        # Spread probability
        if self.spread_model is not None:
            spread_probs = self.spread_model.predict_proba(X)[:, 1]
        else:
            spread_probs = win_probs  # Fallback
        
        # Total prediction
        if self.total_model is not None:
            total_preds = self.total_model.predict(X)
        else:
            total_preds = np.full(len(X), 44.5)  # League average
        
        results = []
        for i in range(len(X)):
            win_prob = float(win_probs[i])
            spread_prob = float(spread_probs[i])
            total_pred = float(total_preds[i])
            
            # Calculate confidence (how far from 50/50)
            confidence = abs(win_prob - 0.5) * 2 * 100  # 0-100 scale
            
            # Generate recommendation
            if win_prob >= 0.65:
                recommendation = "STRONG_BET"
            elif win_prob >= 0.58:
                recommendation = "BET"
            elif win_prob >= 0.52:
                recommendation = "LEAN"
            elif win_prob <= 0.35:
                recommendation = "STRONG_FADE"
            elif win_prob <= 0.42:
                recommendation = "FADE"
            else:
                recommendation = "NO_BET"
            
            results.append({
                'win_probability': win_prob,
                'spread_probability': spread_prob,
                'predicted_total': total_pred,
                'confidence': confidence,
                'recommendation': recommendation,
            })
        
        return results[0] if len(results) == 1 else results
    
    def _heuristic_predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Fallback heuristic when model not trained"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        results = []
        for row in X:
            # Use spread line as primary indicator (index 0)
            spread = row[0] if len(row) > 0 else 0
            implied_prob = row[2] if len(row) > 2 else 0.5
            
            # Spread-based probability
            if spread != 0:
                win_prob = 0.5 - (spread * 0.03)
            else:
                win_prob = implied_prob if implied_prob > 0 else 0.5
            
            win_prob = max(0.2, min(0.8, win_prob))
            
            confidence = abs(win_prob - 0.5) * 2 * 100
            
            if win_prob >= 0.60:
                recommendation = "BET"
            elif win_prob >= 0.52:
                recommendation = "LEAN"
            else:
                recommendation = "NO_BET"
            
            results.append({
                'win_probability': win_prob,
                'spread_probability': win_prob,
                'predicted_total': 44.5,
                'confidence': confidence,
                'recommendation': recommendation,
            })
        
        return results[0] if len(results) == 1 else results
    
    def save(self, path: str = "model"):
        """Save model to disk"""
        os.makedirs(path, exist_ok=True)
        
        if self.win_model:
            joblib.dump(self.win_model, f"{path}/win_model.joblib")
        if self.spread_model:
            joblib.dump(self.spread_model, f"{path}/spread_model.joblib")
        if self.total_model:
            joblib.dump(self.total_model, f"{path}/total_model.joblib")
        
        metadata = {
            'is_trained': self.is_trained,
            'training_accuracy': self.training_accuracy,
            'cv_accuracy': self.cv_accuracy,
            'feature_importance': self.feature_importance,
        }
        joblib.dump(metadata, f"{path}/metadata.joblib")
        
        print(f"Model saved to {path}/")
    
    def load(self, path: str = "model") -> bool:
        """Load model from disk"""
        try:
            if os.path.exists(f"{path}/win_model.joblib"):
                self.win_model = joblib.load(f"{path}/win_model.joblib")
            if os.path.exists(f"{path}/spread_model.joblib"):
                self.spread_model = joblib.load(f"{path}/spread_model.joblib")
            if os.path.exists(f"{path}/total_model.joblib"):
                self.total_model = joblib.load(f"{path}/total_model.joblib")
            if os.path.exists(f"{path}/metadata.joblib"):
                metadata = joblib.load(f"{path}/metadata.joblib")
                self.is_trained = metadata.get('is_trained', False)
                self.training_accuracy = metadata.get('training_accuracy', 0)
                self.cv_accuracy = metadata.get('cv_accuracy', 0)
                self.feature_importance = metadata.get('feature_importance', {})
            
            print(f"Model loaded from {path}/")
            print(f"  Training accuracy: {self.training_accuracy:.4f}")
            print(f"  CV accuracy: {self.cv_accuracy:.4f}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics"""
        return {
            'is_trained': self.is_trained,
            'training_accuracy': self.training_accuracy,
            'cv_accuracy': self.cv_accuracy,
            'feature_importance': self.feature_importance,
            'has_spread_model': self.spread_model is not None,
            'has_total_model': self.total_model is not None,
        }
