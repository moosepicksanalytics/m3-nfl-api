"""
Scraper for NFL historical data from ESPN API.
Used to gather training data for the model.
"""

import httpx
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# ESPN NFL API endpoints
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

# NFL Team ID mapping
NFL_TEAMS = {
    "ARI": {"id": "22", "name": "Arizona Cardinals"},
    "ATL": {"id": "1", "name": "Atlanta Falcons"},
    "BAL": {"id": "33", "name": "Baltimore Ravens"},
    "BUF": {"id": "2", "name": "Buffalo Bills"},
    "CAR": {"id": "29", "name": "Carolina Panthers"},
    "CHI": {"id": "3", "name": "Chicago Bears"},
    "CIN": {"id": "4", "name": "Cincinnati Bengals"},
    "CLE": {"id": "5", "name": "Cleveland Browns"},
    "DAL": {"id": "6", "name": "Dallas Cowboys"},
    "DEN": {"id": "7", "name": "Denver Broncos"},
    "DET": {"id": "8", "name": "Detroit Lions"},
    "GB": {"id": "9", "name": "Green Bay Packers"},
    "HOU": {"id": "34", "name": "Houston Texans"},
    "IND": {"id": "11", "name": "Indianapolis Colts"},
    "JAX": {"id": "30", "name": "Jacksonville Jaguars"},
    "KC": {"id": "12", "name": "Kansas City Chiefs"},
    "LV": {"id": "13", "name": "Las Vegas Raiders"},
    "LAC": {"id": "24", "name": "Los Angeles Chargers"},
    "LAR": {"id": "14", "name": "Los Angeles Rams"},
    "MIA": {"id": "15", "name": "Miami Dolphins"},
    "MIN": {"id": "16", "name": "Minnesota Vikings"},
    "NE": {"id": "17", "name": "New England Patriots"},
    "NO": {"id": "18", "name": "New Orleans Saints"},
    "NYG": {"id": "19", "name": "New York Giants"},
    "NYJ": {"id": "20", "name": "New York Jets"},
    "PHI": {"id": "21", "name": "Philadelphia Eagles"},
    "PIT": {"id": "23", "name": "Pittsburgh Steelers"},
    "SF": {"id": "25", "name": "San Francisco 49ers"},
    "SEA": {"id": "26", "name": "Seattle Seahawks"},
    "TB": {"id": "27", "name": "Tampa Bay Buccaneers"},
    "TEN": {"id": "10", "name": "Tennessee Titans"},
    "WAS": {"id": "28", "name": "Washington Commanders"},
}


async def fetch_standings(season: int = 2024) -> Dict[str, dict]:
    """
    Fetch current NFL standings from ESPN.
    Returns dict keyed by team abbreviation.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{ESPN_BASE}/standings",
            params={"season": season}
        )
        data = response.json()
    
    standings = {}
    
    for group in data.get("children", []):
        for division in group.get("children", []):
            for entry in division.get("standings", {}).get("entries", []):
                team = entry.get("team", {})
                abbrev = team.get("abbreviation", "")
                stats = {s["name"]: s["value"] for s in entry.get("stats", [])}
                
                standings[abbrev] = {
                    "team": abbrev,
                    "name": team.get("displayName", ""),
                    "wins": int(stats.get("wins", 0)),
                    "losses": int(stats.get("losses", 0)),
                    "ties": int(stats.get("ties", 0)),
                    "points_for": int(stats.get("pointsFor", 0)),
                    "points_against": int(stats.get("pointsAgainst", 0)),
                    "point_diff": int(stats.get("pointDifferential", 0)),
                    "streak": stats.get("streak", ""),
                }
    
    return standings


async def fetch_team_record(team_id: str, season: int = 2024) -> dict:
    """
    Fetch detailed team record including home/away splits.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{ESPN_BASE}/teams/{team_id}",
            params={"season": season}
        )
        data = response.json()
    
    team = data.get("team", {})
    record = team.get("record", {}).get("items", [])
    
    overall = next((r for r in record if r.get("type") == "total"), {})
    home = next((r for r in record if r.get("type") == "home"), {})
    away = next((r for r in record if r.get("type") == "road"), {})
    
    def parse_record(rec):
        stats = rec.get("stats", [])
        return {s["name"]: s["value"] for s in stats}
    
    return {
        "overall": parse_record(overall),
        "home": parse_record(home),
        "away": parse_record(away),
    }


async def fetch_completed_games(
    season: int = 2024,
    week: Optional[int] = None
) -> List[dict]:
    """
    Fetch completed NFL games for training data.
    """
    async with httpx.AsyncClient() as client:
        params = {"season": season, "seasontype": 2}  # Regular season
        if week:
            params["week"] = week
        
        response = await client.get(
            f"{ESPN_BASE}/scoreboard",
            params=params
        )
        data = response.json()
    
    games = []
    
    for event in data.get("events", []):
        competition = event.get("competitions", [{}])[0]
        status = competition.get("status", {}).get("type", {})
        
        # Only completed games
        if status.get("completed", False):
            competitors = competition.get("competitors", [])
            
            home = next((c for c in competitors if c.get("homeAway") == "home"), {})
            away = next((c for c in competitors if c.get("homeAway") == "away"), {})
            
            home_score = int(home.get("score", 0))
            away_score = int(away.get("score", 0))
            
            games.append({
                "game_id": event.get("id"),
                "date": event.get("date"),
                "home_team": home.get("team", {}).get("abbreviation"),
                "away_team": away.get("team", {}).get("abbreviation"),
                "home_score": home_score,
                "away_score": away_score,
                "home_won": home_score > away_score,
                "total_points": home_score + away_score,
                "margin": home_score - away_score,
            })
    
    return games


async def fetch_injuries(team_id: str) -> List[dict]:
    """
    Fetch team injuries from ESPN.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{ESPN_BASE}/teams/{team_id}/injuries"
        )
        data = response.json()
    
    injuries = []
    for item in data.get("injuries", []):
        for injury in item.get("injuries", []):
            athlete = injury.get("athlete", {})
            injuries.append({
                "name": athlete.get("displayName"),
                "position": athlete.get("position", {}).get("abbreviation"),
                "status": injury.get("status"),
                "description": injury.get("longComment", ""),
            })
    
    return injuries


async def build_training_dataset(
    seasons: List[int] = [2023, 2024]
) -> List[dict]:
    """
    Build a complete training dataset from historical seasons.
    """
    all_games = []
    
    for season in seasons:
        logger.info(f"Fetching season {season}...")
        
        # Get standings at end of season for team stats
        standings = await fetch_standings(season)
        
        # Fetch all weeks
        for week in range(1, 19):  # Weeks 1-18
            games = await fetch_completed_games(season, week)
            
            for game in games:
                home_abbrev = game["home_team"]
                away_abbrev = game["away_team"]
                
                home_stats = standings.get(home_abbrev, {})
                away_stats = standings.get(away_abbrev, {})
                
                if home_stats and away_stats:
                    training_item = {
                        "home_features": {
                            "team": home_abbrev,
                            "team_full_name": home_stats.get("name", home_abbrev),
                            "wins": home_stats.get("wins", 0),
                            "losses": home_stats.get("losses", 0),
                            "ties": home_stats.get("ties", 0),
                            "games_played": home_stats.get("wins", 0) + home_stats.get("losses", 0),
                            "points_for": home_stats.get("points_for", 0),
                            "points_against": home_stats.get("points_against", 0),
                            "point_diff": home_stats.get("point_diff", 0),
                            "home_wins": 0,
                            "home_losses": 0,
                            "away_wins": 0,
                            "away_losses": 0,
                            "streak_type": None,
                            "streak_length": 0,
                            "last_5_wins": 2,
                            "last_5_losses": 2,
                            "division_rank": 2,
                            "conference_rank": 8,
                            "qb_injured": False,
                            "key_injuries_count": 0,
                        },
                        "away_features": {
                            "team": away_abbrev,
                            "team_full_name": away_stats.get("name", away_abbrev),
                            "wins": away_stats.get("wins", 0),
                            "losses": away_stats.get("losses", 0),
                            "ties": away_stats.get("ties", 0),
                            "games_played": away_stats.get("wins", 0) + away_stats.get("losses", 0),
                            "points_for": away_stats.get("points_for", 0),
                            "points_against": away_stats.get("points_against", 0),
                            "point_diff": away_stats.get("point_diff", 0),
                            "home_wins": 0,
                            "home_losses": 0,
                            "away_wins": 0,
                            "away_losses": 0,
                            "streak_type": None,
                            "streak_length": 0,
                            "last_5_wins": 2,
                            "last_5_losses": 2,
                            "division_rank": 2,
                            "conference_rank": 8,
                            "qb_injured": False,
                            "key_injuries_count": 0,
                        },
                        "home_won": game["home_won"],
                        "total_points": game["total_points"],
                        "margin": game["margin"],
                    }
                    all_games.append(training_item)
            
            await asyncio.sleep(0.5)  # Rate limiting
    
    logger.info(f"Built dataset with {len(all_games)} games")
    return all_games


if __name__ == "__main__":
    # Test fetching current standings
    async def main():
        standings = await fetch_standings()
        print(f"Fetched {len(standings)} teams")
        for abbrev, stats in list(standings.items())[:5]:
            print(f"  {abbrev}: {stats}")
    
    asyncio.run(main())
