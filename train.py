"""
Training script for enhanced NFL prediction model with betting lines
"""
import numpy as np
from model import NFLPredictionModel
from scraper import build_training_dataset
from features import FEATURE_NAMES

def main():
    print("=" * 60)
    print("Enhanced NFL Prediction Model Training")
    print("=" * 60)
    
    # Build dataset with 6 seasons of data
    seasons = [2018, 2019, 2020, 2021, 2022, 2023]
    X, y_win, y_spread, y_total = build_training_dataset(seasons)
    
    if len(X) == 0:
        print("No training data available. Exiting.")
        return
    
    print(f"\nDataset Summary:")
    print(f"  Total samples: {len(X)}")
    print(f"  Features per sample: {len(X[0])}")
    print(f"  Home win rate: {y_win.mean():.3f}")
    print(f"  Home cover rate: {y_spread.mean():.3f}")
    print(f"  Average total points: {y_total.mean():.1f}")
    print(f"  Feature names: {FEATURE_NAMES[:5]}...")
    
    # Initialize and train model
    model = NFLPredictionModel()
    
    print("\nTraining model with hyperparameter tuning...")
    results = model.train(
        X, y_win, 
        y_spread=y_spread,
        y_total=y_total,
        tune_hyperparameters=True
    )
    
    print("\n" + "=" * 60)
    print("Training Results:")
    print("=" * 60)
    print(f"  Win Model Training Accuracy: {results.get('training_accuracy', 0):.4f}")
    print(f"  Win Model CV Accuracy: {results.get('cv_accuracy', 0):.4f}")
    if 'spread_accuracy' in results:
        print(f"  Spread Model Training Accuracy: {results.get('spread_accuracy', 0):.4f}")
    if 'total_mae' in results:
        print(f"  Total Model MAE: {results.get('total_mae', 0):.2f} points")
    
    # Save the model
    model.save("model")
    
    # Test prediction
    print("\n" + "=" * 60)
    print("Test Prediction (using first training sample):")
    print("=" * 60)
    
    test_prediction = model.predict(X[0])
    print(f"  Win Probability: {test_prediction['win_probability']:.3f}")
    print(f"  Spread Probability: {test_prediction['spread_probability']:.3f}")
    print(f"  Predicted Total: {test_prediction['predicted_total']:.1f}")
    print(f"  Confidence: {test_prediction['confidence']:.1f}%")
    print(f"  Recommendation: {test_prediction['recommendation']}")
    
    print("\n" + "=" * 60)
    print("Model training complete! Files saved to model/")
    print("=" * 60)

if __name__ == "__main__":
    main()
