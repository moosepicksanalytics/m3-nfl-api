"""
NFL Prediction Model using Gradient Boosting.
Trains on historical data and makes predictions.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
import os
from typing import Optional, Tuple
import logging

from features import NFLTeamFeatures, calculate_team_strength, calculate_win_probability

logger = logging.getLogger(__name__)

class NFLPredictionModel:
    """
    Gradient Boosting model for NFL game predictions.
    Predicts: Win probability, Spread coverage, Total points
    """
    
    def __init__(self):
        self.win_classifier: Optional[GradientBoostingClassifier] = None
        self.spread_classifier: Optional[GradientBoostingClassifier] = None
        self.total_regressor: Optional[GradientBoostingRegressor] = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_version = "v1.0"
        
    def _extract_features(
        self,
        home: NFLTeamFeatures,
        away: NFLTeamFeatures,
        spread_line: float = 0,
        total_line: float = 45
    ) -> np.ndarray:
        """Extract feature vector for a matchup"""
        home_strength = calculate_team_strength(home, is_home=True)
        away_strength = calculate_team_strength(away, is_home=False)
        
        features = [
            # Team strengths
            home_strength,
            away_strength,
            home_strength - away_strength,
            
            # Records
            home.win_pct,
            away.win_pct,
            home.win_pct - away.win_pct,
            
            # Point differentials
            home.point_diff,
            away.point_diff,
            home.avg_points_for,
            away.avg_points_for,
            home.avg_points_against,
            away.avg_points_against,
            
            # Recent form
            home.recent_form_score,
            away.recent_form_score,
            home.recent_form_score - away.recent_form_score,
            
            # Home/Away performance
            home.home_win_pct,
            away.away_win_pct,
            
            # Injuries (critical for NFL)
            1.0 if home.qb_injured else 0.0,
            1.0 if away.qb_injured else 0.0,
            home.key_injuries_count,
            away.key_injuries_count,
            
            # Streak momentum
            home.streak_length * (1 if home.streak_type == 'W' else -1),
            away.streak_length * (1 if away.streak_type == 'W' else -1),
            
            # Turnovers
            home.turnover_diff,
            away.turnover_diff,
            
            # Lines for spread/total predictions
            spread_line,
            total_line,
        ]
        
        return np.array(features).reshape(1, -1)
    
    def train(
        self,
        training_data: list[dict],
        min_samples: int = 100
    ) -> dict:
        """
        Train the model on historical game data.
        
        training_data format:
        [
            {
                "home_features": {...},
                "away_features": {...},
                "home_won": True/False,
                "home_covered": True/False,
                "total_points": 45,
                "spread_line": -3.5,
                "total_line": 44.5
            },
            ...
        ]
        """
        if len(training_data) < min_samples:
            logger.warning(f"Insufficient training data: {len(training_data)} < {min_samples}")
            return {
                "success": False,
                "message": f"Need at least {min_samples} samples"
            }
        
        # Prepare feature matrices
        X_list = []
        y_win = []
        y_spread = []
        y_total = []
        
        for game in training_data:
            try:
                home = NFLTeamFeatures(**game["home_features"])
                away = NFLTeamFeatures(**game["away_features"])
                
                features = self._extract_features(
                    home, away,
                    game.get("spread_line", 0),
                    game.get("total_line", 45)
                )
                
                X_list.append(features.flatten())
                y_win.append(1 if game["home_won"] else 0)
                y_spread.append(1 if game.get("home_covered", False) else 0)
                y_total.append(game.get("total_points", 45))
                
            except Exception as e:
                logger.warning(f"Skipping invalid game data: {e}")
                continue
        
        if len(X_list) < min_samples:
            return {"success": False, "message": "Too many invalid samples"}
        
        X = np.array(X_list)
        y_win = np.array(y_win)
        y_spread = np.array(y_spread)
        y_total = np.array(y_total)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train win classifier
        self.win_classifier = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        self.win_classifier.fit(X_scaled, y_win)
        win_cv = cross_val_score(self.win_classifier, X_scaled, y_win, cv=5)
        
        # Train spread classifier
        self.spread_classifier = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        self.spread_classifier.fit(X_scaled, y_spread)
        spread_cv = cross_val_score(self.spread_classifier, X_scaled, y_spread, cv=5)
        
        # Train total regressor
        self.total_regressor = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        self.total_regressor.fit(X_scaled, y_total)
        
        self.is_trained = True
        
        return {
            "success": True,
            "samples": len(X_list),
            "win_accuracy": float(np.mean(win_cv)),
            "spread_accuracy": float(np.mean(spread_cv)),
            "total_mae": float(np.mean(np.abs(
                self.total_regressor.predict(X_scaled) - y_total
            )))
        }
    
    def predict(
        self,
        home: NFLTeamFeatures,
        away: NFLTeamFeatures,
        spread_line: float = 0,
        total_line: float = 45
    ) -> dict:
        """
        Make predictions for a matchup.
        Returns probabilities and recommendations.
        """
        # Calculate base strength scores
        home_strength = calculate_team_strength(home, is_home=True)
        away_strength = calculate_team_strength(away, is_home=False)
        
        # Heuristic fallback if model not trained
        if not self.is_trained or self.win_classifier is None:
            home_prob, away_prob = calculate_win_probability(
                home_strength, away_strength, home, away
            )
            
            return {
                "home_team": home.team,
                "away_team": away.team,
                "home_strength": round(home_strength, 1),
                "away_strength": round(away_strength, 1),
                "home_win_prob": round(home_prob * 100, 1),
                "away_win_prob": round(away_prob * 100, 1),
                "predicted_total": round(
                    (home.avg_points_for + away.avg_points_for + 
                     home.avg_points_against + away.avg_points_against) / 2, 1
                ),
                "confidence": 70.0,
                "recommendation": self._get_recommendation(
                    home_prob, home_strength - away_strength
                ),
                "model_used": "heuristic"
            }
        
        # Extract features
        X = self._extract_features(home, away, spread_line, total_line)
        X_scaled = self.scaler.transform(X)
        
        # Get ML predictions
        win_proba = self.win_classifier.predict_proba(X_scaled)[0]
        spread_proba = self.spread_classifier.predict_proba(X_scaled)[0]
        predicted_total = self.total_regressor.predict(X_scaled)[0]
        
        home_win_prob = win_proba[1]  # Probability of class 1 (home win)
        home_cover_prob = spread_proba[1] if len(spread_proba) > 1 else 0.5
        
        # Blend ML with heuristic (70% ML, 30% heuristic for robustness)
        heuristic_prob, _ = calculate_win_probability(
            home_strength, away_strength, home, away
        )
        blended_prob = 0.7 * home_win_prob + 0.3 * heuristic_prob
        
        # Calculate confidence based on probability distance from 0.5
        confidence = 70 + abs(blended_prob - 0.5) * 50
        
        return {
            "home_team": home.team,
            "away_team": away.team,
            "home_strength": round(home_strength, 1),
            "away_strength": round(away_strength, 1),
            "home_win_prob": round(blended_prob * 100, 1),
            "away_win_prob": round((1 - blended_prob) * 100, 1),
            "home_cover_prob": round(home_cover_prob * 100, 1),
            "predicted_total": round(predicted_total, 1),
            "confidence": round(min(95, confidence), 1),
            "recommendation": self._get_recommendation(
                blended_prob, home_strength - away_strength
            ),
            "model_used": "gradient_boosting",
            "feature_breakdown": {
                "home_form": round(home.recent_form_score * 100, 0),
                "away_form": round(away.recent_form_score * 100, 0),
                "home_qb_out": home.qb_injured,
                "away_qb_out": away.qb_injured,
            }
        }
    
    def _get_recommendation(self, home_prob: float, strength_diff: float) -> str:
        """Generate recommendation tier"""
        edge = abs(home_prob - 0.5)
        
        if edge >= 0.20:
            return "STRONG"
        elif edge >= 0.12:
            return "GOOD"
        elif edge >= 0.05:
            return "FAIR"
        else:
            return "WEAK"
    
    def save(self, path: str = "model"):
        """Save trained model to disk"""
        if not os.path.exists(path):
            os.makedirs(path)
        
        if self.win_classifier:
            joblib.dump(self.win_classifier, f"{path}/win_classifier.joblib")
        if self.spread_classifier:
            joblib.dump(self.spread_classifier, f"{path}/spread_classifier.joblib")
        if self.total_regressor:
            joblib.dump(self.total_regressor, f"{path}/total_regressor.joblib")
        joblib.dump(self.scaler, f"{path}/scaler.joblib")
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str = "model") -> bool:
        """Load trained model from disk"""
        try:
            self.win_classifier = joblib.load(f"{path}/win_classifier.joblib")
            self.spread_classifier = joblib.load(f"{path}/spread_classifier.joblib")
            self.total_regressor = joblib.load(f"{path}/total_regressor.joblib")
            self.scaler = joblib.load(f"{path}/scaler.joblib")
            self.is_trained = True
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
            return False
