"""
Script to train the NFL model with historical data.
Run this before deploying to Railway.
"""

import asyncio
import json
import logging
from scraper import build_training_dataset
from model import NFLPredictionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    logger.info("🏈 NFL M3 Model Training")
    logger.info("=" * 50)
    
    # Step 1: Fetch historical data
    logger.info("📊 Fetching historical NFL data...")
    training_data = await build_training_dataset(seasons=[2023, 2024])
    
    # Save training data for reference
    with open("training_data.json", "w") as f:
        json.dump(training_data, f, indent=2)
    logger.info(f"✓ Saved {len(training_data)} games to training_data.json")
    
    # Step 2: Train the model
    logger.info("🤖 Training model...")
    model = NFLPredictionModel()
    result = model.train(training_data)
    
    if result["success"]:
        logger.info(f"✓ Training complete!")
        logger.info(f"  Samples: {result['samples']}")
        logger.info(f"  Win Accuracy: {result['win_accuracy']:.1%}")
        logger.info(f"  Spread Accuracy: {result['spread_accuracy']:.1%}")
        logger.info(f"  Total MAE: {result['total_mae']:.1f} points")
        
        # Save the model
        model.save("model")
        logger.info("✓ Model saved to ./model/")
    else:
        logger.error(f"Training failed: {result['message']}")


if __name__ == "__main__":
    asyncio.run(main())
