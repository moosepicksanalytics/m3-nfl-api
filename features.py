"""
Enhanced NFL Feature Engineering with Betting Lines
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class NFLTeamFeatures:
    """Enhanced NFL team features including betting lines"""
    # Basic record
    wins: int = 0
    losses: int = 0
    ties: int = 0
    
    # Scoring
    points_for: float = 0.0
    points_against: float = 0.0
    
    # Advanced metrics
    point_differential: float = 0.0
    yards_per_game: float = 0.0
    yards_allowed_per_game: float = 0.0
    turnover_diff: float = 0.0
    
    # Situational
    home_wins: int = 0
    home_losses: int = 0
    away_wins: int = 0
    away_losses: int = 0
    
    # Recent form
    last_5_wins: int = 0
    last_5_losses: int = 0
    streak: int = 0  # Positive = win streak, negative = loss streak
    
    # Rest and scheduling
    days_rest: int = 7
    is_divisional: bool = False
    is_primetime: bool = False
    
    # Betting lines (market consensus)
    spread_line: float = 0.0
    total_line: float = 0.0
    moneyline: int = 0
    implied_prob: float = 0.5
    
    # Opponent strength
    opponent_win_pct: float = 0.5
    strength_of_schedule: float = 0.5

# Updated feature weights with betting line importance
NFL_FEATURE_WEIGHTS = {
    # Betting lines are MOST predictive (market wisdom)
    'spread_line': 0.20,          # Vegas spread is highly predictive
    'implied_prob': 0.12,         # Moneyline implied probability
    'total_line': 0.05,           # Over/under context
    
    # Team performance
    'win_percentage': 0.12,
    'point_differential': 0.10,
    'recent_form': 0.10,          # Last 5 games
    
    # Situational factors
    'home_advantage': 0.08,
    'days_rest': 0.06,
    'opponent_strength': 0.07,
    
    # Advanced metrics
    'turnover_diff': 0.05,
    'yards_diff': 0.03,
    'divisional_game': 0.02,
}

def moneyline_to_implied_prob(ml: int) -> float:
    """Convert American moneyline to implied probability"""
    if ml is None or np.isnan(ml):
        return 0.5
    if ml > 0:
        return 100 / (ml + 100)
    else:
        return abs(ml) / (abs(ml) + 100)

def calculate_team_strength(features: NFLTeamFeatures) -> float:
    """Calculate overall team strength score (0-100)"""
    games_played = features.wins + features.losses + features.ties
    if games_played == 0:
        return 50.0
    
    # Win percentage (0-30 points)
    win_pct = features.wins / games_played
    win_score = win_pct * 30
    
    # Point differential per game (0-25 points)
    ppg_diff = features.point_differential / games_played
    # Normalize: +15 ppg diff = max score, -15 = min score
    pd_score = max(0, min(25, (ppg_diff + 15) / 30 * 25))
    
    # Recent form - last 5 games (0-20 points)
    recent_games = features.last_5_wins + features.last_5_losses
    if recent_games > 0:
        recent_win_pct = features.last_5_wins / recent_games
        form_score = recent_win_pct * 20
    else:
        form_score = 10
    
    # Streak bonus/penalty (0-10 points, centered at 5)
    streak_score = 5 + min(5, max(-5, features.streak))
    
    # Turnover differential (0-10 points)
    to_score = max(0, min(10, (features.turnover_diff + 10) / 20 * 10))
    
    # Home/away performance (0-5 points)
    if features.home_wins + features.home_losses > 0:
        home_pct = features.home_wins / (features.home_wins + features.home_losses)
    else:
        home_pct = 0.5
    home_score = home_pct * 5
    
    total = win_score + pd_score + form_score + streak_score + to_score + home_score
    return min(100, max(0, total))

def calculate_win_probability(
    home_features: NFLTeamFeatures,
    away_features: NFLTeamFeatures,
    use_betting_lines: bool = True
) -> float:
    """
    Calculate home team win probability using team features and betting lines.
    Betting lines are heavily weighted when available.
    """
    # Start with team strength comparison
    home_strength = calculate_team_strength(home_features)
    away_strength = calculate_team_strength(away_features)
    
    # Strength-based probability
    strength_diff = home_strength - away_strength
    strength_prob = 0.5 + (strength_diff / 100) * 0.35  # Max swing of 35%
    
    # Home field advantage (+3% base)
    home_advantage = 0.03
    
    # Rest advantage
    rest_diff = home_features.days_rest - away_features.days_rest
    rest_advantage = rest_diff * 0.005  # 0.5% per day of rest advantage
    
    # Recent form adjustment
    home_form = home_features.last_5_wins - home_features.last_5_losses
    away_form = away_features.last_5_wins - away_features.last_5_losses
    form_diff = home_form - away_form
    form_adjustment = form_diff * 0.015  # 1.5% per win differential
    
    # Calculate base probability
    base_prob = strength_prob + home_advantage + rest_advantage + form_adjustment
    
    # If betting lines available, weight them heavily (60% betting, 40% model)
    if use_betting_lines and home_features.implied_prob > 0:
        betting_prob = home_features.implied_prob
        # Blend: 60% betting line, 40% model
        final_prob = (betting_prob * 0.6) + (base_prob * 0.4)
    else:
        final_prob = base_prob
    
    # Clamp to reasonable range
    return max(0.15, min(0.85, final_prob))

def calculate_spread_probability(
    home_features: NFLTeamFeatures,
    away_features: NFLTeamFeatures,
    spread: float
) -> float:
    """Calculate probability of home team covering the spread"""
    # Get base win probability
    win_prob = calculate_win_probability(home_features, away_features)
    
    # Adjust for spread
    # Each point of spread shifts probability by ~3%
    spread_adjustment = spread * 0.03
    
    cover_prob = win_prob + spread_adjustment
    
    return max(0.20, min(0.80, cover_prob))

def calculate_predicted_total(
    home_features: NFLTeamFeatures,
    away_features: NFLTeamFeatures
) -> float:
    """Predict total points scored in game"""
    games_home = home_features.wins + home_features.losses + home_features.ties
    games_away = away_features.wins + away_features.losses + away_features.ties
    
    if games_home == 0 or games_away == 0:
        # Use betting line if available, else league average
        if home_features.total_line > 0:
            return home_features.total_line
        return 44.5  # NFL average
    
    # Calculate expected scoring
    home_ppg = home_features.points_for / games_home
    home_papg = home_features.points_against / games_home
    away_ppg = away_features.points_for / games_away
    away_papg = away_features.points_against / games_away
    
    # Expected points: (Team's offense + Opponent's defense) / 2
    home_expected = (home_ppg + away_papg) / 2
    away_expected = (away_ppg + home_papg) / 2
    
    predicted_total = home_expected + away_expected
    
    # Blend with betting total if available (50/50)
    if home_features.total_line > 0:
        predicted_total = (predicted_total * 0.5) + (home_features.total_line * 0.5)
    
    return predicted_total

def prepare_features_for_model(
    home_features: NFLTeamFeatures,
    away_features: NFLTeamFeatures
) -> list:
    """
    Prepare feature vector for ML model input.
    Returns list of numeric features in consistent order.
    """
    home_games = home_features.wins + home_features.losses + home_features.ties
    away_games = away_features.wins + away_features.losses + away_features.ties
    
    features = [
        # Betting lines (most important)
        home_features.spread_line,
        home_features.total_line,
        home_features.implied_prob,
        
        # Win percentages
        home_features.wins / max(1, home_games),
        away_features.wins / max(1, away_games),
        
        # Point differentials per game
        home_features.point_differential / max(1, home_games),
        away_features.point_differential / max(1, away_games),
        
        # Points per game
        home_features.points_for / max(1, home_games),
        away_features.points_for / max(1, away_games),
        home_features.points_against / max(1, home_games),
        away_features.points_against / max(1, away_games),
        
        # Recent form
        home_features.last_5_wins,
        home_features.last_5_losses,
        away_features.last_5_wins,
        away_features.last_5_losses,
        
        # Streaks
        home_features.streak,
        away_features.streak,
        
        # Rest
        home_features.days_rest,
        away_features.days_rest,
        
        # Turnover diff
        home_features.turnover_diff,
        away_features.turnover_diff,
        
        # Home/away splits
        home_features.home_wins / max(1, home_features.home_wins + home_features.home_losses),
        away_features.away_wins / max(1, away_features.away_wins + away_features.away_losses),
        
        # Opponent strength
        home_features.opponent_win_pct,
        away_features.opponent_win_pct,
        
        # Situational
        1.0 if home_features.is_divisional else 0.0,
        1.0 if home_features.is_primetime else 0.0,
    ]
    
    return features

FEATURE_NAMES = [
    'spread_line', 'total_line', 'implied_prob',
    'home_win_pct', 'away_win_pct',
    'home_point_diff_pg', 'away_point_diff_pg',
    'home_ppg', 'away_ppg', 'home_papg', 'away_papg',
    'home_last5_wins', 'home_last5_losses', 'away_last5_wins', 'away_last5_losses',
    'home_streak', 'away_streak',
    'home_rest', 'away_rest',
    'home_to_diff', 'away_to_diff',
    'home_home_win_pct', 'away_away_win_pct',
    'home_opp_strength', 'away_opp_strength',
    'is_divisional', 'is_primetime',
]
