"""
FastAPI server for NFL M3 Predictions API.
Endpoints for predictions and training.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import logging

from model import NFLPredictionModel
from features import NFLTeamFeatures

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(
    title="M3 NFL Predictions API",
    description="Machine learning predictions for NFL games",
    version="1.0.0"
)

# CORS for Supabase edge functions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = NFLPredictionModel()

# Try to load existing model
model.load("model")


# === Request/Response Models ===

class TeamFeaturesRequest(BaseModel):
    team: str
    team_full_name: str
    wins: int
    losses: int
    ties: int = 0
    games_played: int
    points_for: int
    points_against: int
    point_diff: int
    home_wins: int = 0
    home_losses: int = 0
    away_wins: int = 0
    away_losses: int = 0
    streak_type: Optional[str] = None
    streak_length: int = 0
    last_5_wins: int = 2
    last_5_losses: int = 2
    division_rank: int = 4
    conference_rank: int = 16
    qb_injured: bool = False
    key_injuries_count: int = 0
    yards_per_game: float = 0.0
    yards_allowed_per_game: float = 0.0
    turnover_diff: int = 0
    third_down_pct: float = 0.0
    red_zone_pct: float = 0.0


class PredictionRequest(BaseModel):
    home_team: TeamFeaturesRequest
    away_team: TeamFeaturesRequest
    spread_line: float = 0
    total_line: float = 45


class BatchPredictionRequest(BaseModel):
    games: List[PredictionRequest]


class TrainingDataItem(BaseModel):
    home_features: dict
    away_features: dict
    home_won: bool
    home_covered: bool = False
    total_points: int = 45
    spread_line: float = 0
    total_line: float = 45


class TrainRequest(BaseModel):
    training_data: List[TrainingDataItem]


# === Endpoints ===

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "service": "M3 NFL Predictions API",
        "status": "healthy",
        "model_trained": model.is_trained,
        "version": model.model_version
    }


@app.get("/health")
def health():
    """Health check for Railway"""
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Get prediction for a single NFL game.
    """
    try:
        home = NFLTeamFeatures(
            team=request.home_team.team,
            team_full_name=request.home_team.team_full_name,
            wins=request.home_team.wins,
            losses=request.home_team.losses,
            ties=request.home_team.ties,
            games_played=request.home_team.games_played,
            points_for=request.home_team.points_for,
            points_against=request.home_team.points_against,
            point_diff=request.home_team.point_diff,
            home_wins=request.home_team.home_wins,
            home_losses=request.home_team.home_losses,
            away_wins=request.home_team.away_wins,
            away_losses=request.home_team.away_losses,
            streak_type=request.home_team.streak_type,
            streak_length=request.home_team.streak_length,
            last_5_wins=request.home_team.last_5_wins,
            last_5_losses=request.home_team.last_5_losses,
            division_rank=request.home_team.division_rank,
            conference_rank=request.home_team.conference_rank,
            qb_injured=request.home_team.qb_injured,
            key_injuries_count=request.home_team.key_injuries_count,
            yards_per_game=request.home_team.yards_per_game,
            yards_allowed_per_game=request.home_team.yards_allowed_per_game,
            turnover_diff=request.home_team.turnover_diff,
        )
        
        away = NFLTeamFeatures(
            team=request.away_team.team,
            team_full_name=request.away_team.team_full_name,
            wins=request.away_team.wins,
            losses=request.away_team.losses,
            ties=request.away_team.ties,
            games_played=request.away_team.games_played,
            points_for=request.away_team.points_for,
            points_against=request.away_team.points_against,
            point_diff=request.away_team.point_diff,
            home_wins=request.away_team.home_wins,
            home_losses=request.away_team.home_losses,
            away_wins=request.away_team.away_wins,
            away_losses=request.away_team.away_losses,
            streak_type=request.away_team.streak_type,
            streak_length=request.away_team.streak_length,
            last_5_wins=request.away_team.last_5_wins,
            last_5_losses=request.away_team.last_5_losses,
            division_rank=request.away_team.division_rank,
            conference_rank=request.away_team.conference_rank,
            qb_injured=request.away_team.qb_injured,
            key_injuries_count=request.away_team.key_injuries_count,
            yards_per_game=request.away_team.yards_per_game,
            yards_allowed_per_game=request.away_team.yards_allowed_per_game,
            turnover_diff=request.away_team.turnover_diff,
        )
        
        prediction = model.predict(
            home, away,
            request.spread_line,
            request.total_line
        )
        
        return prediction
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    """
    Get predictions for multiple NFL games.
    """
    predictions = []
    for game in request.games:
        try:
            result = predict(game)
            predictions.append(result)
        except Exception as e:
            predictions.append({"error": str(e)})
    
    return {"predictions": predictions}


@app.post("/train")
def train(request: TrainRequest):
    """
    Train the model with historical data.
    """
    try:
        training_data = [item.model_dump() for item in request.training_data]
        result = model.train(training_data)
        
        if result["success"]:
            model.save("model")
        
        return result
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
def model_info():
    """Get information about the current model"""
    return {
        "is_trained": model.is_trained,
        "version": model.model_version,
        "model_type": "GradientBoosting" if model.is_trained else "Heuristic"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
