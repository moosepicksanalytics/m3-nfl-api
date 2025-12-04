"""
NFL-specific feature engineering for M3 predictions.
Football features differ significantly from hockey.
"""

from dataclasses import dataclass
from typing import Optional
import math

@dataclass
class NFLTeamFeatures:
    """NFL team features for prediction model"""
    team: str
    team_full_name: str
    
    # Record
    wins: int
    losses: int
    ties: int
    games_played: int
    
    # Scoring
    points_for: int
    points_against: int
    point_diff: int
    
    # Home/Away splits
    home_wins: int
    home_losses: int
    away_wins: int
    away_losses: int
    
    # Recent form
    streak_type: Optional[str]  # 'W' or 'L'
    streak_length: int
    last_5_wins: int
    last_5_losses: int
    
    # Division/Conference
    division_rank: int
    conference_rank: int
    
    # Injury impact
    qb_injured: bool  # Starting QB out
    key_injuries_count: int  # Starters on IR
    
    # Advanced (optional)
    yards_per_game: float = 0.0
    yards_allowed_per_game: float = 0.0
    turnover_diff: int = 0
    third_down_pct: float = 0.0
    red_zone_pct: float = 0.0
    
    @property
    def win_pct(self) -> float:
        if self.games_played == 0:
            return 0.5
        return self.wins / self.games_played
    
    @property
    def avg_points_for(self) -> float:
        if self.games_played == 0:
            return 21.0  # NFL average
        return self.points_for / self.games_played
    
    @property
    def avg_points_against(self) -> float:
        if self.games_played == 0:
            return 21.0
        return self.points_against / self.games_played
    
    @property
    def home_win_pct(self) -> float:
        home_games = self.home_wins + self.home_losses
        if home_games == 0:
            return 0.5
        return self.home_wins / home_games
    
    @property
    def away_win_pct(self) -> float:
        away_games = self.away_wins + self.away_losses
        if away_games == 0:
            return 0.5
        return self.away_wins / away_games
    
    @property
    def recent_form_score(self) -> float:
        """Score from 0-1 based on last 5 games"""
        return self.last_5_wins / 5.0
    
    def to_feature_dict(self) -> dict:
        """Convert to dictionary for model input"""
        return {
            "team": self.team,
            "wins": self.wins,
            "losses": self.losses,
            "games_played": self.games_played,
            "win_pct": self.win_pct,
            "points_for": self.points_for,
            "points_against": self.points_against,
            "point_diff": self.point_diff,
            "avg_points_for": self.avg_points_for,
            "avg_points_against": self.avg_points_against,
            "home_wins": self.home_wins,
            "home_losses": self.home_losses,
            "away_wins": self.away_wins,
            "away_losses": self.away_losses,
            "home_win_pct": self.home_win_pct,
            "away_win_pct": self.away_win_pct,
            "streak_type": self.streak_type,
            "streak_length": self.streak_length,
            "last_5_wins": self.last_5_wins,
            "recent_form": self.recent_form_score,
            "division_rank": self.division_rank,
            "conference_rank": self.conference_rank,
            "qb_injured": self.qb_injured,
            "key_injuries_count": self.key_injuries_count,
            "yards_per_game": self.yards_per_game,
            "yards_allowed_per_game": self.yards_allowed_per_game,
            "turnover_diff": self.turnover_diff,
        }


# NFL-specific feature weights (tuned for football)
NFL_FEATURE_WEIGHTS = {
    "win_pct": 0.22,           # Overall record matters
    "point_diff": 0.18,        # Point differential is key
    "recent_form": 0.15,       # Last 5 games momentum
    "home_advantage": 0.12,    # Home field ~3 points in NFL
    "qb_health": 0.15,         # QB is most important position
    "turnover_diff": 0.10,     # Turnovers swing games
    "yards_diff": 0.08,        # Offensive/defensive efficiency
}


def calculate_team_strength(features: NFLTeamFeatures, is_home: bool) -> float:
    """
    Calculate overall team strength score (0-100).
    Higher = stronger team.
    """
    score = 50.0  # Base score
    
    # Win percentage impact (0.22 weight, max ±15 points)
    win_pct_impact = (features.win_pct - 0.5) * 30 * NFL_FEATURE_WEIGHTS["win_pct"] / 0.22
    score += win_pct_impact
    
    # Point differential impact (0.18 weight, max ±12 points)
    # NFL point diff of +100 over 17 games = +5.9 per game = elite
    ppg_diff = features.point_diff / max(features.games_played, 1)
    point_diff_impact = (ppg_diff / 10) * 24 * NFL_FEATURE_WEIGHTS["point_diff"] / 0.18
    score += max(min(point_diff_impact, 12), -12)
    
    # Recent form impact (0.15 weight, max ±8 points)
    form_impact = (features.recent_form_score - 0.5) * 16 * NFL_FEATURE_WEIGHTS["recent_form"] / 0.15
    score += form_impact
    
    # Home advantage (0.12 weight, +3 points)
    if is_home:
        score += 3.0 * NFL_FEATURE_WEIGHTS["home_advantage"] / 0.12
    
    # QB injury impact (0.15 weight, -10 points if starting QB out)
    if features.qb_injured:
        score -= 10.0 * NFL_FEATURE_WEIGHTS["qb_health"] / 0.15
    
    # Key injuries impact (max -5 for 5+ starters out)
    injury_penalty = min(features.key_injuries_count, 5) * 1.0
    score -= injury_penalty
    
    # Turnover differential impact (0.10 weight, max ±5 points)
    to_impact = (features.turnover_diff / 10) * 10 * NFL_FEATURE_WEIGHTS["turnover_diff"] / 0.10
    score += max(min(to_impact, 5), -5)
    
    # Clamp to valid range
    return max(20.0, min(80.0, score))


def calculate_win_probability(
    home_strength: float,
    away_strength: float,
    home_features: NFLTeamFeatures,
    away_features: NFLTeamFeatures
) -> tuple[float, float]:
    """
    Calculate win probability for home and away teams.
    Returns (home_win_prob, away_win_prob).
    """
    # Base probability from strength differential
    strength_diff = home_strength - away_strength
    
    # Convert to probability using logistic function
    # 10 point diff = ~65% win probability
    k = 0.08  # Steepness factor
    home_prob = 1 / (1 + math.exp(-k * strength_diff))
    
    # Adjust for specific matchup factors
    
    # Divisional games are more unpredictable
    # (would need division data to implement)
    
    # Weather/outdoor factor (would need weather data)
    
    # Ensure probabilities are reasonable
    home_prob = max(0.20, min(0.80, home_prob))
    away_prob = 1 - home_prob
    
    return home_prob, away_prob


def calculate_predicted_total(
    home_features: NFLTeamFeatures,
    away_features: NFLTeamFeatures
) -> float:
    """
    Predict total points for the game.
    NFL average is ~43-46 points per game.
    """
    # Base: average of both teams' scoring
    home_ppg = home_features.avg_points_for
    away_ppg = away_features.avg_points_for
    home_papg = home_features.avg_points_against
    away_papg = away_features.avg_points_against
    
    # Matchup-based projection
    home_expected = (home_ppg + away_papg) / 2
    away_expected = (away_ppg + home_papg) / 2
    base_total = home_expected + away_expected
    
    # Adjustments
    adjustments = 0.0
    
    # Hot offenses add points
    if home_features.recent_form_score >= 0.8:
        adjustments += 1.5
    if away_features.recent_form_score >= 0.8:
        adjustments += 1.5
    
    # Cold offenses subtract
    if home_features.recent_form_score <= 0.2:
        adjustments -= 1.5
    if away_features.recent_form_score <= 0.2:
        adjustments -= 1.5
    
    # QB injuries = fewer points for that team
    if home_features.qb_injured:
        adjustments -= 3.0
    if away_features.qb_injured:
        adjustments -= 3.0
    
    return base_total + adjustments


def calculate_spread_probability(
    home_strength: float,
    away_strength: float,
    spread_line: float,
    home_features: NFLTeamFeatures,
    away_features: NFLTeamFeatures
) -> float:
    """
    Calculate probability of home team covering the spread.
    spread_line is from home team perspective (negative = favorite).
    """
    # Expected margin = strength difference
    # In NFL, 1 point of strength ≈ 0.3 expected margin points
    expected_margin = (home_strength - away_strength) * 0.3
    
    # Home team covers if actual margin > spread_line
    # spread_line of -7 means home must win by >7
    # spread_line of +3 means home can lose by <3
    
    # Standard deviation of NFL games is about 13-14 points
    std_dev = 13.5
    
    # Z-score: how many SDs is spread from expected margin
    z = (expected_margin - spread_line) / std_dev
    
    # Convert to probability using normal CDF approximation
    # (simplified logistic approximation)
    cover_prob = 1 / (1 + math.exp(-0.6 * z))
    
    return max(0.25, min(0.75, cover_prob))
