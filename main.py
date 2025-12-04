"""
FastAPI server for NFL prediction model
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
from model import NFLPredictionModel
from features import NFLTeamFeatures, prepare_features_for_model, moneyline_to_implied_prob

app = FastAPI(
    title="M3 NFL Prediction API",
    description="Enhanced NFL game prediction model with betting line features",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
model = NFLPredictionModel()
model.load("model")

class TeamInput(BaseModel):
    """Input for a single team's features"""
    team: str
    wins: int = 0
    losses: int = 0
    ties: int = 0
    points_for: float = 0
    points_against: float = 0
    home_wins: int = 0
    home_losses: int = 0
    away_wins: int = 0
    away_losses: int = 0
    last_5_wins: int = 0
    last_5_losses: int = 0
    streak: int = 0
    days_rest: int = 7
    is_divisional: bool = False
    is_primetime: bool = False
    # Betting lines
    spread_line: float = 0
    total_line: float = 44.5
    moneyline: int = 0
    opponent_win_pct: float = 0.5
    turnover_diff: float = 0

class PredictionRequest(BaseModel):
    """Request for game prediction"""
    home_team: TeamInput
    away_team: TeamInput

class BatchPredictionRequest(BaseModel):
    """Request for multiple game predictions"""
    games: List[PredictionRequest]

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "M3 NFL Prediction API v2.0",
        "model_trained": model.is_trained,
        "training_accuracy": model.training_accuracy,
        "cv_accuracy": model.cv_accuracy,
    }

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model.is_trained}

@app.post("/predict")
async def predict(request: PredictionRequest) -> Dict[str, Any]:
    """Make prediction for a single game"""
    try:
        # Build feature objects
        home_features = NFLTeamFeatures(
            wins=request.home_team.wins,
            losses=request.home_team.losses,
            ties=request.home_team.ties,
            points_for=request.home_team.points_for,
            points_against=request.home_team.points_against,
            point_differential=request.home_team.points_for - request.home_team.points_against,
            home_wins=request.home_team.home_wins,
            home_losses=request.home_team.home_losses,
            away_wins=request.home_team.away_wins,
            away_losses=request.home_team.away_losses,
            last_5_wins=request.home_team.last_5_wins,
            last_5_losses=request.home_team.last_5_losses,
            streak=request.home_team.streak,
            days_rest=request.home_team.days_rest,
            is_divisional=request.home_team.is_divisional,
            is_primetime=request.home_team.is_primetime,
            spread_line=request.home_team.spread_line,
            total_line=request.home_team.total_line,
            moneyline=request.home_team.moneyline,
            implied_prob=moneyline_to_implied_prob(request.home_team.moneyline),
            opponent_win_pct=request.home_team.opponent_win_pct,
            turnover_diff=request.home_team.turnover_diff,
        )
        
        away_features = NFLTeamFeatures(
            wins=request.away_team.wins,
            losses=request.away_team.losses,
            ties=request.away_team.ties,
            points_for=request.away_team.points_for,
            points_against=request.away_team.points_against,
            point_differential=request.away_team.points_for - request.away_team.points_against,
            home_wins=request.away_team.home_wins,
            home_losses=request.away_team.home_losses,
            away_wins=request.away_team.away_wins,
            away_losses=request.away_team.away_losses,
            last_5_wins=request.away_team.last_5_wins,
            last_5_losses=request.away_team.last_5_losses,
            streak=request.away_team.streak,
            days_rest=request.away_team.days_rest,
            is_divisional=request.away_team.is_divisional,
            is_primetime=request.away_team.is_primetime,
            spread_line=-request.home_team.spread_line,  # Flip spread for away
            total_line=request.home_team.total_line,
            moneyline=request.away_team.moneyline,
            implied_prob=moneyline_to_implied_prob(request.away_team.moneyline),
            opponent_win_pct=request.away_team.opponent_win_pct,
            turnover_diff=request.away_team.turnover_diff,
        )
        
        # Prepare features and predict
        features = prepare_features_for_model(home_features, away_features)
        prediction = model.predict(np.array(features))
        
        return {
            "home_team": request.home_team.team,
            "away_team": request.away_team.team,
            **prediction
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest) -> List[Dict[str, Any]]:
    """Make predictions for multiple games"""
    results = []
    for game in request.games:
        result = await predict(game)
        results.append(result)
    return results

@app.post("/train")
async def train_model():
    """Trigger model retraining"""
    try:
        from scraper import build_training_dataset
        
        X, y_win, y_spread, y_total = build_training_dataset(
            seasons=[2018, 2019, 2020, 2021, 2022, 2023]
        )
        
        if len(X) == 0:
            raise HTTPException(status_code=500, detail="No training data available")
        
        results = model.train(X, y_win, y_spread=y_spread, y_total=y_total)
        model.save("model")
        
        return {
            "status": "success",
            "samples": len(X),
            "training_accuracy": results.get('training_accuracy'),
            "cv_accuracy": results.get('cv_accuracy'),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    """Get model information"""
    return model.get_model_info()
