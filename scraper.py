"""
Enhanced NFL Data Scraper using nfl_data_py with betting lines
"""
import nfl_data_py as nfl
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
from features import NFLTeamFeatures, moneyline_to_implied_prob

# NFL team abbreviation mappings
TEAM_ABBREV_MAP = {
    'ARI': 'ARI', 'ATL': 'ATL', 'BAL': 'BAL', 'BUF': 'BUF',
    'CAR': 'CAR', 'CHI': 'CHI', 'CIN': 'CIN', 'CLE': 'CLE',
    'DAL': 'DAL', 'DEN': 'DEN', 'DET': 'DET', 'GB': 'GB',
    'HOU': 'HOU', 'IND': 'IND', 'JAX': 'JAX', 'KC': 'KC',
    'LAC': 'LAC', 'LA': 'LA', 'LAR': 'LA', 'LV': 'LV', 'OAK': 'LV',
    'MIA': 'MIA', 'MIN': 'MIN', 'NE': 'NE', 'NO': 'NO',
    'NYG': 'NYG', 'NYJ': 'NYJ', 'PHI': 'PHI', 'PIT': 'PIT',
    'SEA': 'SEA', 'SF': 'SF', 'TB': 'TB', 'TEN': 'TEN',
    'WAS': 'WAS', 'WSH': 'WAS',
    'SD': 'LAC', 'STL': 'LA', 'OAK': 'LV',
}

def normalize_team(team: str) -> str:
    """Normalize team abbreviation"""
    return TEAM_ABBREV_MAP.get(team, team)

def fetch_schedule_with_odds(seasons: List[int]) -> pd.DataFrame:
    """
    Fetch NFL schedules with betting lines from nfl_data_py.
    Returns DataFrame with games and betting odds.
    """
    print(f"Fetching schedules for seasons: {seasons}")
    
    try:
        schedules = nfl.import_schedules(seasons)
        print(f"Fetched {len(schedules)} games")
        
        # Available betting columns:
        # spread_line, total_line, away_moneyline, home_moneyline,
        # away_spread_odds, home_spread_odds, under_odds, over_odds
        
        # Filter to regular season and completed games
        schedules = schedules[schedules['game_type'] == 'REG']
        schedules = schedules[schedules['home_score'].notna()]
        schedules = schedules[schedules['away_score'].notna()]
        
        # Normalize team names
        schedules['home_team'] = schedules['home_team'].apply(normalize_team)
        schedules['away_team'] = schedules['away_team'].apply(normalize_team)
        
        print(f"After filtering: {len(schedules)} completed regular season games")
        return schedules
        
    except Exception as e:
        print(f"Error fetching schedules: {e}")
        return pd.DataFrame()

def build_training_dataset(
    seasons: List[int] = [2018, 2019, 2020, 2021, 2022, 2023]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build complete training dataset with betting line features.
    
    Returns:
        X: Feature matrix
        y_win: Win labels (1 = home win)
        y_spread: Spread cover labels (1 = home covered)
        y_total: Total points scored
    """
    print(f"Building training dataset for seasons: {seasons}")
    
    schedules = fetch_schedule_with_odds(seasons)
    
    if schedules.empty:
        print("No schedule data available")
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Sort by date to build running records
    schedules = schedules.sort_values(['season', 'week']).reset_index(drop=True)
    
    # Track running team stats per season
    team_stats = {}
    
    X = []
    y_win = []
    y_spread = []
    y_total = []
    
    for idx, game in schedules.iterrows():
        season = game['season']
        week = game['week']
        home_team = game['home_team']
        away_team = game['away_team']
        
        # Initialize season stats if needed
        season_key = f"{season}"
        if season_key not in team_stats:
            team_stats[season_key] = {}
        
        for team in [home_team, away_team]:
            if team not in team_stats[season_key]:
                team_stats[season_key][team] = {
                    'wins': 0, 'losses': 0, 'ties': 0,
                    'points_for': 0, 'points_against': 0,
                    'home_wins': 0, 'home_losses': 0,
                    'away_wins': 0, 'away_losses': 0,
                    'last_5_results': [],
                    'streak': 0,
                    'turnover_diff': 0,
                }
        
        home_stats = team_stats[season_key][home_team]
        away_stats = team_stats[season_key][away_team]
        
        # Require minimum games played for training data
        home_games = home_stats['wins'] + home_stats['losses'] + home_stats['ties']
        away_games = away_stats['wins'] + away_stats['losses'] + away_stats['ties']
        
        if home_games >= 2 and away_games >= 2:
            # Extract betting lines
            spread_line = game.get('spread_line', 0) or 0
            total_line = game.get('total_line', 44.5) or 44.5
            home_ml = game.get('home_moneyline', 0) or 0
            away_ml = game.get('away_moneyline', 0) or 0
            
            # Calculate implied probabilities
            home_implied = moneyline_to_implied_prob(home_ml)
            away_implied = moneyline_to_implied_prob(away_ml)
            
            # Build home features
            home_features = NFLTeamFeatures(
                wins=home_stats['wins'],
                losses=home_stats['losses'],
                ties=home_stats['ties'],
                points_for=home_stats['points_for'],
                points_against=home_stats['points_against'],
                point_differential=home_stats['points_for'] - home_stats['points_against'],
                home_wins=home_stats['home_wins'],
                home_losses=home_stats['home_losses'],
                away_wins=home_stats['away_wins'],
                away_losses=home_stats['away_losses'],
                last_5_wins=sum(1 for r in home_stats['last_5_results'][-5:] if r == 'W'),
                last_5_losses=sum(1 for r in home_stats['last_5_results'][-5:] if r == 'L'),
                streak=home_stats['streak'],
                days_rest=game.get('home_rest', 7) or 7,
                is_divisional=game.get('div_game', False) or False,
                spread_line=spread_line,
                total_line=total_line,
                moneyline=home_ml,
                implied_prob=home_implied,
                opponent_win_pct=away_stats['wins'] / max(1, away_games),
                turnover_diff=home_stats['turnover_diff'],
            )
            
            # Build away features
            away_features = NFLTeamFeatures(
                wins=away_stats['wins'],
                losses=away_stats['losses'],
                ties=away_stats['ties'],
                points_for=away_stats['points_for'],
                points_against=away_stats['points_against'],
                point_differential=away_stats['points_for'] - away_stats['points_against'],
                home_wins=away_stats['home_wins'],
                home_losses=away_stats['home_losses'],
                away_wins=away_stats['away_wins'],
                away_losses=away_stats['away_losses'],
                last_5_wins=sum(1 for r in away_stats['last_5_results'][-5:] if r == 'W'),
                last_5_losses=sum(1 for r in away_stats['last_5_results'][-5:] if r == 'L'),
                streak=away_stats['streak'],
                days_rest=game.get('away_rest', 7) or 7,
                spread_line=-spread_line,  # Flip for away
                total_line=total_line,
                moneyline=away_ml,
                implied_prob=away_implied,
                opponent_win_pct=home_stats['wins'] / max(1, home_games),
                turnover_diff=away_stats['turnover_diff'],
            )
            
            # Prepare features
            from features import prepare_features_for_model
            features = prepare_features_for_model(home_features, away_features)
            
            # Labels
            home_score = game['home_score']
            away_score = game['away_score']
            
            home_win = 1 if home_score > away_score else 0
            
            # Spread cover: home team covers if (home_score - away_score) > -spread_line
            # spread_line is negative when home is favored (e.g., -7)
            home_margin = home_score - away_score
            home_covered = 1 if home_margin > -spread_line else 0
            
            total_points = home_score + away_score
            
            X.append(features)
            y_win.append(home_win)
            y_spread.append(home_covered)
            y_total.append(total_points)
        
        # Update running stats after the game
        home_score = game['home_score']
        away_score = game['away_score']
        
        # Home team updates
        if home_score > away_score:
            home_stats['wins'] += 1
            home_stats['home_wins'] += 1
            home_stats['last_5_results'].append('W')
            home_stats['streak'] = home_stats['streak'] + 1 if home_stats['streak'] > 0 else 1
        elif home_score < away_score:
            home_stats['losses'] += 1
            home_stats['home_losses'] += 1
            home_stats['last_5_results'].append('L')
            home_stats['streak'] = home_stats['streak'] - 1 if home_stats['streak'] < 0 else -1
        else:
            home_stats['ties'] += 1
            home_stats['last_5_results'].append('T')
            home_stats['streak'] = 0
        
        home_stats['points_for'] += home_score
        home_stats['points_against'] += away_score
        
        # Away team updates
        if away_score > home_score:
            away_stats['wins'] += 1
            away_stats['away_wins'] += 1
            away_stats['last_5_results'].append('W')
            away_stats['streak'] = away_stats['streak'] + 1 if away_stats['streak'] > 0 else 1
        elif away_score < home_score:
            away_stats['losses'] += 1
            away_stats['away_losses'] += 1
            away_stats['last_5_results'].append('L')
            away_stats['streak'] = away_stats['streak'] - 1 if away_stats['streak'] < 0 else -1
        else:
            away_stats['ties'] += 1
            away_stats['last_5_results'].append('T')
            away_stats['streak'] = 0
        
        away_stats['points_for'] += away_score
        away_stats['points_against'] += home_score
    
    print(f"Built dataset with {len(X)} training samples")
    
    return (
        np.array(X),
        np.array(y_win),
        np.array(y_spread),
        np.array(y_total)
    )

def fetch_current_standings() -> Dict[str, Dict]:
    """
    Fetch current NFL standings for live predictions.
    Uses most recent season data.
    """
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # NFL season runs Sept-Feb
    if current_month < 3:
        season = current_year - 1
    elif current_month >= 9:
        season = current_year
    else:
        season = current_year - 1  # Offseason, use last year
    
    try:
        schedules = nfl.import_schedules([season])
        schedules = schedules[schedules['game_type'] == 'REG']
        schedules = schedules[schedules['home_score'].notna()]
        schedules = schedules.sort_values('week')
        
        standings = {}
        
        for _, game in schedules.iterrows():
            home = normalize_team(game['home_team'])
            away = normalize_team(game['away_team'])
            
            for team in [home, away]:
                if team not in standings:
                    standings[team] = {
                        'wins': 0, 'losses': 0, 'ties': 0,
                        'points_for': 0, 'points_against': 0,
                        'home_wins': 0, 'home_losses': 0,
                        'away_wins': 0, 'away_losses': 0,
                        'last_5': [],
                        'streak': 0,
                    }
            
            home_score = game['home_score']
            away_score = game['away_score']
            
            # Update home team
            standings[home]['points_for'] += home_score
            standings[home]['points_against'] += away_score
            if home_score > away_score:
                standings[home]['wins'] += 1
                standings[home]['home_wins'] += 1
                standings[home]['last_5'].append('W')
            elif home_score < away_score:
                standings[home]['losses'] += 1
                standings[home]['home_losses'] += 1
                standings[home]['last_5'].append('L')
            else:
                standings[home]['ties'] += 1
                standings[home]['last_5'].append('T')
            
            # Update away team
            standings[away]['points_for'] += away_score
            standings[away]['points_against'] += home_score
            if away_score > home_score:
                standings[away]['wins'] += 1
                standings[away]['away_wins'] += 1
                standings[away]['last_5'].append('W')
            elif away_score < home_score:
                standings[away]['losses'] += 1
                standings[away]['away_losses'] += 1
                standings[away]['last_5'].append('L')
            else:
                standings[away]['ties'] += 1
                standings[away]['last_5'].append('T')
        
        # Calculate streaks
        for team, stats in standings.items():
            last_5 = stats['last_5'][-5:]
            stats['last_5_wins'] = sum(1 for r in last_5 if r == 'W')
            stats['last_5_losses'] = sum(1 for r in last_5 if r == 'L')
            
            # Calculate current streak
            if last_5:
                streak = 0
                last_result = last_5[-1]
                for r in reversed(last_5):
                    if r == last_result:
                        streak += 1 if last_result == 'W' else -1 if last_result == 'L' else 0
                    else:
                        break
                stats['streak'] = streak
        
        return standings
        
    except Exception as e:
        print(f"Error fetching standings: {e}")
        return {}
