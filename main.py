
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple, Union, Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("football_stats.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MATCH_COUNT = 20
DATA_DIR = "football_data"
CONFIG_FILE = "config.json"
API_KEY_ENV_VAR = "FOOTBALL_API_KEY"

# Configuration
class Config:
    """Configuration manager for the application"""
    
    def __init__(self):
        """Initialize configuration with default values"""
        self.api_key = os.environ.get(API_KEY_ENV_VAR, "")
        self.api_base_url = "https://api.football-data.org/v4"
        self.cache_expiry = 24  # hours
        self.match_count = DEFAULT_MATCH_COUNT
        self.load_config()
        
    def load_config(self):
        """Load configuration from file if it exists"""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    config_data = json.load(f)
                    self.api_key = config_data.get('api_key', self.api_key)
                    self.api_base_url = config_data.get('api_base_url', self.api_base_url)
                    self.cache_expiry = config_data.get('cache_expiry', self.cache_expiry)
                    self.match_count = config_data.get('match_count', self.match_count)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            config_data = {
                'api_key': self.api_key,
                'api_base_url': self.api_base_url,
                'cache_expiry': self.cache_expiry,
                'match_count': self.match_count
            }
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config_data, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")


# Data Collection
class DataCollector:
    """Handles data collection from the football API"""
    
    def __init__(self, config: Config):
        """Initialize data collector with configuration"""
        self.config = config
        self.headers = {'X-Auth-Token': self.config.api_key}
        self.ensure_data_dir()
        
    def ensure_data_dir(self):
        """Ensure data directory exists"""
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
    
    def get_team_id(self, team_name: str) -> Optional[int]:
        """
        Get team ID from team name
        
        Args:
            team_name: Name of the team to search for
            
        Returns:
            Team ID if found, None otherwise
        """
        try:
            cache_file = os.path.join(DATA_DIR, "teams_cache.json")
            
            # Check if cache exists and is fresh
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    if datetime.now().timestamp() - cache_data.get('timestamp', 0) < self.config.cache_expiry * 3600:
                        teams = cache_data.get('teams', [])
                        for team in teams:
                            if team_name.lower() in team.get('name', '').lower():
                                return team.get('id')
            
            # If not in cache or cache expired, fetch from API
            response = requests.get(
                f"{self.config.api_base_url}/teams",
                headers=self.headers
            )
            response.raise_for_status()
            
            teams_data = response.json().get('teams', [])
            
            # Update cache
            cache_data = {
                'timestamp': datetime.now().timestamp(),
                'teams': teams_data
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            # Search for team
            for team in teams_data:
                if team_name.lower() in team.get('name', '').lower():
                    return team.get('id')
            
            return None
        except Exception as e:
            logger.error(f"Error getting team ID: {e}")
            return None
    
    def get_matches(self, team_id: int, limit: int = None) -> List[Dict]:
        """
        Get matches for a team
        
        Args:
            team_id: ID of the team
            limit: Maximum number of matches to retrieve
            
        Returns:
            List of match data dictionaries
        """
        if limit is None:
            limit = self.config.match_count
            
        try:
            cache_file = os.path.join(DATA_DIR, f"matches_{team_id}.json")
            
            # Check if cache exists and is fresh
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    if datetime.now().timestamp() - cache_data.get('timestamp', 0) < self.config.cache_expiry * 3600:
                        return cache_data.get('matches', [])[:limit]
            
            # If not in cache or cache expired, fetch from API
            # First get finished matches
            response = requests.get(
                f"{self.config.api_base_url}/teams/{team_id}/matches?status=FINISHED",
                headers=self.headers
            )
            response.raise_for_status()
            
            matches_data = response.json().get('matches', [])
            
            # Update cache
            cache_data = {
                'timestamp': datetime.now().timestamp(),
                'matches': matches_data
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            return matches_data[:limit]
        except Exception as e:
            logger.error(f"Error getting matches: {e}")
            return []
    
    def get_match_statistics(self, match_id: int) -> Dict:
        """
        Get detailed statistics for a match
        
        Args:
            match_id: ID of the match
            
        Returns:
            Dictionary containing match statistics
        """
        try:
            cache_file = os.path.join(DATA_DIR, f"match_stats_{match_id}.json")
            
            # Check if cache exists and is fresh
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    if datetime.now().timestamp() - cache_data.get('timestamp', 0) < self.config.cache_expiry * 3600:
                        return cache_data.get('statistics', {})
            
            # If not in cache or cache expired, fetch from API
            response = requests.get(
                f"{self.config.api_base_url}/matches/{match_id}",
                headers=self.headers
            )
            response.raise_for_status()
            
            match_data = response.json()
            
            # Extract statistics
            statistics = {
                'match_id': match_id,
                'date': match_data.get('utcDate'),
                'home_team': match_data.get('homeTeam', {}).get('name'),
                'away_team': match_data.get('awayTeam', {}).get('name'),
                'home_score': match_data.get('score', {}).get('fullTime', {}).get('home'),
                'away_score': match_data.get('score', {}).get('fullTime', {}).get('away'),
                'home_fouls': 0,
                'away_fouls': 0,
                'home_shots_on_target': 0,
                'away_shots_on_target': 0,
                'home_yellow_cards': 0,
                'away_yellow_cards': 0,
                'home_red_cards': 0,
                'away_red_cards': 0,
                'referee': match_data.get('referees', [{}])[0].get('name', 'Unknown')
            }
            
            # Get detailed statistics if available
            if 'statistics' in match_data:
                for stat in match_data['statistics']:
                    if stat.get('type') == 'FOULS':
                        statistics['home_fouls'] = stat.get('home', 0)
                        statistics['away_fouls'] = stat.get('away', 0)
                    elif stat.get('type') == 'SHOTS_ON_TARGET':
                        statistics['home_shots_on_target'] = stat.get('home', 0)
                        statistics['away_shots_on_target'] = stat.get('away', 0)
            
            # Count cards
            for card in match_data.get('bookings', []):
                team = card.get('team', {}).get('name')
                card_type = card.get('card')
                
                if team == statistics['home_team']:
                    if card_type == 'YELLOW':
                        statistics['home_yellow_cards'] += 1
                    elif card_type == 'RED':
                        statistics['home_red_cards'] += 1
                elif team == statistics['away_team']:
                    if card_type == 'YELLOW':
                        statistics['away_yellow_cards'] += 1
                    elif card_type == 'RED':
                        statistics['away_red_cards'] += 1
            
            # Update cache
            cache_data = {
                'timestamp': datetime.now().timestamp(),
                'statistics': statistics
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            return statistics
        except Exception as e:
            logger.error(f"Error getting match statistics: {e}")
            return {}


# Data Analysis
class DataAnalyzer:
    """Analyzes collected football statistics"""
    
    def __init__(self, collector: DataCollector):
        """Initialize analyzer with data collector"""
        self.collector = collector
    
    def get_team_statistics(self, team_name: str, match_count: int = None) -> Dict:
        """
        Get comprehensive statistics for a team
        
        Args:
            team_name: Name of the team
            match_count: Number of matches to analyze
            
        Returns:
            Dictionary containing analyzed statistics
        """
        if match_count is None:
            match_count = self.collector.config.match_count
        
        team_id = self.collector.get_team_id(team_name)
        if not team_id:
            logger.error(f"Team not found: {team_name}")
            return {}
        
        matches = self.collector.get_matches(team_id, match_count)
        if not matches:
            logger.error(f"No matches found for team: {team_name}")
            return {}
        
        # Process match statistics
        match_stats = []
        for match in matches:
            match_id = match.get('id')
            stats = self.collector.get_match_statistics(match_id)
            if stats:
                match_stats.append(stats)
            
            # Rate limiting - sleep to avoid hitting API limits
            time.sleep(0.5)
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(match_stats)
        
        # Filter for matches where the team participated
        team_matches = df[(df['home_team'] == team_name) | (df['away_team'] == team_name)]
        
        # Calculate statistics
        stats = {}
        
        # Total matches analyzed
        stats['matches_analyzed'] = len(team_matches)
        
        # Fouls
        home_fouls = team_matches[team_matches['home_team'] == team_name]['home_fouls'].sum()
        away_fouls = team_matches[team_matches['away_team'] == team_name]['away_fouls'].sum()
        stats['total_fouls'] = home_fouls + away_fouls
        stats['avg_fouls_per_match'] = stats['total_fouls'] / stats['matches_analyzed'] if stats['matches_analyzed'] > 0 else 0
        
        # Shots on target
        home_shots = team_matches[team_matches['home_team'] == team_name]['home_shots_on_target'].sum()
        away_shots = team_matches[team_matches['away_team'] == team_name]['away_shots_on_target'].sum()
        stats['total_shots_on_target'] = home_shots + away_shots
        stats['avg_shots_on_target'] = stats['total_shots_on_target'] / stats['matches_analyzed'] if stats['matches_analyzed'] > 0 else 0
        
        # Yellow cards
        home_yellows = team_matches[team_matches['home_team'] == team_name]['home_yellow_cards'].sum()
        away_yellows = team_matches[team_matches['away_team'] == team_name]['away_yellow_cards'].sum()
        stats['total_yellow_cards'] = home_yellows + away_yellows
        stats['avg_yellow_cards'] = stats['total_yellow_cards'] / stats['matches_analyzed'] if stats['matches_analyzed'] > 0 else 0
        
        # Red cards
        home_reds = team_matches[team_matches['home_team'] == team_name]['home_red_cards'].sum()
        away_reds = team_matches[team_matches['away_team'] == team_name]['away_red_cards'].sum()
        stats['total_red_cards'] = home_reds + away_reds
        stats['avg_red_cards'] = stats['total_red_cards'] / stats['matches_analyzed'] if stats['matches_analyzed'] > 0 else 0
        
        # Referee statistics
        referee_stats = team_matches.groupby('referee').size().reset_index(name='count')
        referee_stats = referee_stats.sort_values('count', ascending=False)
        stats['most_common_referees'] = referee_stats.head(3).to_dict('records')
        
        # Referee card analysis
        referee_cards = []
        for referee in referee_stats.head(5)['referee']:
            ref_matches = team_matches[team_matches['referee'] == referee]
            
            home_matches = ref_matches[ref_matches['home_team'] == team_name]
            away_matches = ref_matches[ref_matches['away_team'] == team_name]
            
            home_yellows = home_matches['home_yellow_cards'].sum()
            away_yellows = away_matches['away_yellow_cards'].sum()
            
            home_reds = home_matches['home_red_cards'].sum()
            away_reds = away_matches['away_red_cards'].sum()
            
            referee_cards.append({
                'referee': referee,
                'matches': len(ref_matches),
                'yellow_cards': home_yellows + away_yellows,
                'red_cards': home_reds + away_
                'red_cards': home_reds + away_reds,
                'cards_per_match': (home_yellows + away_yellows + home_reds + away_reds) / len(ref_matches) if len(ref_matches) > 0 else 0
            })
            
        stats['referee_card_analysis'] = referee_cards
        
        # Match results
        home_wins = team_matches[(team_matches['home_team'] == team_name) & (team_matches['home_score'] > team_matches['away_score'])].shape[0]
        away_wins = team_matches[(team_matches['away_team'] == team_name) & (team_matches['away_score'] > team_matches['home_score'])].shape[0]
        home_draws = team_matches[(team_matches['home_team'] == team_name) & (team_matches['home_score'] == team_matches['away_score'])].shape[0]
        away_draws = team_matches[(team_matches['away_team'] == team_name) & (team_matches['home_score'] == team_matches['away_score'])].shape[0]
        
        stats['wins'] = home_wins + away_wins
        stats['draws'] = home_draws + away_draws
        stats['losses'] = stats['matches_analyzed'] - stats['wins'] - stats['draws']
        stats['win_percentage'] = (stats['wins'] / stats['matches_analyzed'] * 100) if stats['matches_analyzed'] > 0 else 0
        
        # Store raw data for plotting
        stats['raw_data'] = team_matches.to_dict('records')
        
        return stats

    def generate_comparison(self, team1_name: str, team2_name: str, match_count: int = None) -> Dict:
        """
        Generate a comparison between two teams
        
        Args:
            team1_name: Name of the first team
            team2_name: Name of the second team
            match_count: Number of matches to analyze
            
        Returns:
            Dictionary containing comparative statistics
        """
        team1_stats = self.get_team_statistics(team1_name, match_count)
        team2_stats = self.get_team_statistics(team2_name, match_count)
        
        if not team1_stats or not team2_stats:
            return {}
        
        comparison = {
            'team1': team1_name,
            'team2': team2_name,
            'matches_analyzed': {
                'team1': team1_stats.get('matches_analyzed', 0),
                'team2': team2_stats.get('matches_analyzed', 0)
            },
            'fouls': {
                'team1': team1_stats.get('avg_fouls_per_match', 0),
                'team2': team2_stats.get('avg_fouls_per_match', 0)
            },
            'shots_on_target': {
                'team1': team1_stats.get('avg_shots_on_target', 0),
                'team2': team2_stats.get('avg_shots_on_target', 0)
            },
            'yellow_cards': {
                'team1': team1_stats.get('avg_yellow_cards', 0),
                'team2': team2_stats.get('avg_yellow_cards', 0)
            },
            'red_cards': {
                'team1': team1_stats.get('avg_red_cards', 0),
                'team2': team2_stats.get('avg_red_cards', 0)
            },
            'win_percentage': {
                'team1': team1_stats.get('win_percentage', 0),
                'team2': team2_stats.get('win_percentage', 0)
            }
        }
        
        return comparison


# Visualization
class Visualizer:
    """Visualizes football statistics"""
    
    def __init__(self):
        """Initialize visualizer"""
        # Set default style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def plot_team_performance(self, stats: Dict, save_path: str = None):
        """
        Create a visualization of team performance
        
        Args:
            stats: Team statistics dictionary
            save_path: Path to save the visualization (optional)
        """
        if not stats:
            logger.error("No statistics available for visualization")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Team Performance Analysis", fontsize=16)
        
        # Plot match results
        labels = ['Wins', 'Draws', 'Losses']
        sizes = [stats.get('wins', 0), stats.get('draws', 0), stats.get('losses', 0)]
        axes[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=['green', 'gray', 'red'])
        axes[0, 0].set_title('Match Results')
        
        # Plot cards
        card_data = {
            'Yellow Cards': stats.get('avg_yellow_cards', 0),
            'Red Cards': stats.get('avg_red_cards', 0),
        }
        axes[0, 1].bar(card_data.keys(), card_data.values(), color=['gold', 'red'])
        axes[0, 1].set_title('Average Cards per Match')
        axes[0, 1].set_ylabel('Average Count')
        
        # Plot fouls and shots
        other_stats = {
            'Fouls': stats.get('avg_fouls_per_match', 0),
            'Shots on Target': stats.get('avg_shots_on_target', 0),
        }
        axes[1, 0].bar(other_stats.keys(), other_stats.values(), color=['blue', 'orange'])
        axes[1, 0].set_title('Average Match Statistics')
        axes[1, 0].set_ylabel('Average Count')
        
        # Plot referee data if available
        ref_data = stats.get('referee_card_analysis', [])
        if ref_data:
            # Get top 3 referees by cards per match
            ref_data = sorted(ref_data, key=lambda x: x.get('cards_per_match', 0), reverse=True)[:3]
            refs = [r.get('referee', 'Unknown').split()[-1] for r in ref_data]  # Get last names only
            cards_per_match = [r.get('cards_per_match', 0) for r in ref_data]
            
            axes[1, 1].bar(refs, cards_per_match, color='purple')
            axes[1, 1].set_title('Cards per Match by Referee')
            axes[1, 1].set_ylabel('Cards per Match')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No referee data available', horizontalalignment='center', verticalalignment='center')
            axes[1, 1].set_title('Referee Analysis')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()
    
    def plot_team_comparison(self, comparison: Dict, save_path: str = None):
        """
        Create a visualization comparing two teams
        
        Args:
            comparison: Comparison dictionary
            save_path: Path to save the visualization (optional)
        """
        if not comparison:
            logger.error("No comparison data available for visualization")
            return
        
        team1 = comparison.get('team1', 'Team 1')
        team2 = comparison.get('team2', 'Team 2')
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Team Comparison: {team1} vs {team2}", fontsize=16)
        
        # Plot fouls
        fouls = comparison.get('fouls', {})
        axes[0, 0].bar([team1, team2], [fouls.get('team1', 0), fouls.get('team2', 0)], color=['blue', 'red'])
        axes[0, 0].set_title('Average Fouls per Match')
        axes[0, 0].set_ylabel('Count')
        
        # Plot shots on target
        shots = comparison.get('shots_on_target', {})
        axes[0, 1].bar([team1, team2], [shots.get('team1', 0), shots.get('team2', 0)], color=['blue', 'red'])
        axes[0, 1].set_title('Average Shots on Target per Match')
        axes[0, 1].set_ylabel('Count')
        
        # Plot cards
        yellows = comparison.get('yellow_cards', {})
        reds = comparison.get('red_cards', {})
        
        x = np.arange(2)
        width = 0.35
        
        axes[1, 0].bar(x - width/2, [yellows.get('team1', 0), yellows.get('team2', 0)], width, label='Yellow Cards', color='gold')
        axes[1, 0].bar(x + width/2, [reds.get('team1', 0), reds.get('team2', 0)], width, label='Red Cards', color='darkred')
        axes[1, 0].set_title('Average Cards per Match')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([team1, team2])
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend()
        
        # Plot win percentage
        win_pct = comparison.get('win_percentage', {})
        axes[1, 1].bar([team1, team2], [win_pct.get('team1', 0), win_pct.get('team2', 0)], color=['blue', 'red'])
        axes[1, 1].set_title('Win Percentage')
        axes[1, 1].set_ylabel('Percentage')
        axes[1, 1].set_ylim(0, 100)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Comparison visualization saved to {save_path}")
        else:
            plt.show()


# User Interface
class UserInterface:
    """Command-line interface for the application"""
    
    def __init__(self, config: Config, analyzer: DataAnalyzer, visualizer: Visualizer):
        """Initialize UI with configuration and analyzer"""
        self.config = config
        self.analyzer = analyzer
        self.visualizer = visualizer
    
    def setup_api_key(self):
        """Set up API key if not already configured"""
        if not self.config.api_key:
            print("\nAPI Key Setup")
            print("=============")
            print("To use this application, you need a Football Data API key.")
            print("You can get a free API key from https://www.football-data.org/")
            
            api_key = input("\nEnter your API key: ")
            if api_key.strip():
                self.config.api_key = api_key
                self.config.save_config()
                print("API key saved successfully!")
            else:
                print("No API key provided. Some functionality may be limited.")
    
    def display_menu(self):
        """Display main menu"""
        print("\nFootball Statistics Analyzer")
        print("============================")
        print("1. Analyze a team")
        print("2. Compare two teams")
        print("3. Configure settings")
        print("4. Exit")
        
        choice = input("\nSelect an option (1-4): ")
        return choice
    
    def run(self):
        """Run the UI loop"""
        self.setup_api_key()
        
        while True:
            choice = self.display_menu()
            
            if choice == '1':
                self.analyze_team()
            elif choice == '2':
                self.compare_teams()
            elif choice == '3':
                self.configure_settings()
            elif choice == '4':
                print("Thank you for using Football Statistics Analyzer!")
                break
            else:
                print("Invalid choice. Please try again.")
    
    def analyze_team(self):
        """Analyze a single team"""
        print("\nTeam Analysis")
        print("=============")
        
        team_name = input("Enter team name: ")
        if not team_name.strip():
            print("No team name provided.")
            return
        
        print(f"\nAnalyzing {team_name}...")
        stats = self.analyzer.get_team_statistics(team_name)
        
        if not stats:
            print(f"Could not find or analyze team: {team_name}")
            return
        
        self.display_team_stats(stats, team_name)
        
        visualize = input("\nWould you like to see a visualization? (y/n): ").lower()
        if visualize == 'y':
            print("Generating visualization...")
            self.visualizer.plot_team_performance(stats)
    
    def compare_teams(self):
        """Compare two teams"""
        print("\nTeam Comparison")
        print("==============")
        
        team1 = input("Enter first team name: ")
        team2 = input("Enter second team name: ")
        
        if not team1.strip() or not team2.strip():
            print("Both team names are required.")
            return
        
        print(f"\nComparing {team1} and {team2}...")
        comparison = self.analyzer.generate_comparison(team1, team2)
        
        if not comparison:
            print(f"Could not compare teams: {team1} and {team2}")
            return
        
        self.display_comparison(comparison)
        
        visualize = input("\nWould you like to see a visualization? (y/n): ").lower()
        if visualize == 'y':
            print("Generating visualization...")
            self.visualizer.plot_team_comparison(comparison)
    
    def configure_settings(self):
        """Configure application settings"""
        print("\nSettings Configuration")
        print("=====================")
        print(f"1. API Key (current: {'*' * 8})")
        print(f"2. Match Count (current: {self.config.match_count})")
        print(f"3. Cache Expiry (current: {self.config.cache_expiry} hours)")
        print("4. Back to Main Menu")
        
        choice = input("\nSelect an option (1-4): ")
        
        if choice == '1':
            new_key = input("Enter new API key (leave empty to keep current): ")
            if new_key.strip():
                self.config.api_key = new_key
                self.config.save_config()
                print("API key updated successfully!")
        elif choice == '2':
            try:
                new_count = int(input(f"Enter new match count (current: {self.config.match_count}): "))
                if new_count > 0:
                    self.config.match_count = new_count
                    self.config.save_config()
                    print(f"Match count updated to {new_count}!")
                else:
                    print("Match count must be positive.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        elif choice == '3':
            try:
                new_expiry = int(input(f"Enter new cache expiry in hours (current: {self.config.cache_exp
