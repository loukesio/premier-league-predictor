# premier-league-predictor


# Premier League Match Predictor 🔮⚽

A machine learning system for predicting English Premier League match outcomes using historical data and advanced feature engineering.

## Features

- **Data Collection**: Automatically downloads match data from multiple Premier League seasons
- **Advanced Feature Engineering**: 17+ sophisticated features including team strength, form, attack/defense ratings
- **Multiple ML Models**: Random Forest, Gradient Boosting, and Logistic Regression with automated model selection
- **Cross-Validation**: Robust model evaluation with k-fold cross-validation
- **Backtesting**: Test model performance on historical matches
- **Gameweek Predictions**: Predict multiple matches at once
- **Model Persistence**: Save and load trained models
- **Confidence Scoring**: Get probability estimates for each outcome

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/premier-league-predictor.git
cd premier-league-predictor
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the predictor:
```bash
python improved_predictor.py
```

## Requirements

```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
requests>=2.28.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## Usage

### Basic Usage

```python
from improved_predictor import ImprovedPLPredictor

# Initialize predictor
predictor = ImprovedPLPredictor()

# Load and prepare data
predictor.get_football_data()

# Train models (compares multiple algorithms)
predictor.train_models()

# Predict a match
prediction, probabilities = predictor.predict_match('Liverpool', 'Manchester City')
```

### Advanced Usage

```python
# Predict multiple matches
fixtures = [
    ('Arsenal', 'Chelsea'),
    ('Manchester United', 'Tottenham'),
    ('Brighton', 'Newcastle')
]
predictions = predictor.predict_gameweek(fixtures)

# Backtest model performance
accuracy = predictor.backtest_predictions(test_matches=100)

# Save/load trained model
predictor.save_model('my_model.pkl')
predictor.load_model('my_model.pkl')
```

## Model Features

The predictor uses 17 engineered features:

### Team Strength Metrics
- `home_team_strength` / `away_team_strength`: Overall team performance rating
- `home_attack_strength` / `away_attack_strength`: Offensive capability rating
- `home_defense_strength` / `away_defense_strength`: Defensive capability rating

### Form & Performance
- `home_recent_form` / `away_recent_form`: Points from recent matches
- `home_goals_avg` / `away_goals_avg`: Average goals scored per game
- `home_goals_conceded_avg` / `away_goals_conceded_avg`: Average goals conceded
- `home_win_rate` / `away_win_rate`: Win percentage in recent matches

### Advanced Metrics
- `home_advantage`: Home field advantage factor
- `goal_difference_trend`: Relative goal difference between teams
- `head_to_head_record`: Historical performance between specific teams

## Data Sources

The system automatically downloads data from [Football-Data.co.uk](https://www.football-data.co.uk/), including:
- Match results from 2020-21 to current season
- Goals, shots, and other match statistics
- Team names and fixture information

## Performance

Typical model performance:
- **Accuracy**: 52-58% (significantly better than random 33.3%)
- **Cross-validation**: 5-fold CV with stratified sampling
- **Feature Importance**: Attack/defense strength and recent form are typically most predictive

## Model Comparison

The system automatically trains and compares:

1. **Random Forest**: Best for handling non-linear relationships
2. **Gradient Boosting**: Often highest accuracy
3. **Logistic Regression**: Good baseline with interpretable coefficients

## File Structure

```
premier-league-predictor/
├── improved_predictor.py      # Main improved predictor class
├── original_predictor.py      # Original reconstructed code
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── models/                   # Saved model files
│   └── pl_predictor_model.pkl
└── examples/                 # Example usage scripts
    └── example_usage.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Improvements Over Original

The improved version includes:

- **Better Data Handling**: More robust error handling and data validation
- **Advanced Features**: 17 vs 5 features, including head-to-head records
- **Model Selection**: Automated comparison of multiple algorithms
- **Cross-Validation**: Proper model evaluation with CV
- **Backtesting**: Historical performance testing
- **Model Persistence**: Save/load functionality
- **Better Evaluation**: Detailed metrics and feature importance
- **Code Quality**: Better documentation, error handling, and modularity

## Limitations

- **Inherent Unpredictability**: Football matches have high randomness
- **Data Dependency**: Performance depends on data quality and availability
- **Feature Engineering**: May miss important factors (injuries, motivation, etc.)
- **Sample Size**: Limited historical data for some team matchups

## Future Enhancements

- Integration with live APIs for real-time data
- Player-level statistics and injury data
- Weather and stadium condition factors
- Betting odds as additional features
- Deep learning models (LSTM, Transformer)
- Ensemble methods combining multiple approaches

## License

MIT License - see LICENSE file for details.

## Disclaimer

This tool is for educational and research purposes. Sports betting involves risk and this model's predictions should not be used as the sole basis for financial decisions.

## Contact

- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

**Built with Python, scikit-learn, and a passion for football analytics! ⚽📊**


```


"""
Example usage of the Premier League Predictor
This script demonstrates various ways to use the improved predictor
"""

from improved_predictor import ImprovedPLPredictor
import pandas as pd

def basic_example():
    """Basic usage example"""
    print("=== BASIC USAGE EXAMPLE ===")
    
    # Initialize predictor
    predictor = ImprovedPLPredictor()
    
    # Load data
    if not predictor.get_football_data():
        print("Failed to load data")
        return
    
    # Train models
    if not predictor.train_models():
        print("Failed to train models")
        return
    
    # Predict a single match
    prediction, probabilities = predictor.predict_match('Liverpool', 'Manchester City')
    
    return predictor

def advanced_example():
    """Advanced usage with multiple predictions and backtesting"""
    print("\n=== ADVANCED USAGE EXAMPLE ===")
    
    predictor = ImprovedPLPredictor()
    
    # Load data with custom seasons
    custom_seasons = {
        '2022-23': 'https://www.football-data.co.uk/mmz4281/2223/E0.csv',
        '2023-24': 'https://www.football-data.co.uk/mmz4281/2324/E0.csv',
        '2024-25': 'https://www.football-data.co.uk/mmz4281/2425/E0.csv'
    }
    
    if not predictor.get_football_data(custom_seasons):
        return
    
    if not predictor.train_models(cv_folds=3):  # Faster training
        return
    
    # Predict multiple matches
    fixtures = [
        ('Arsenal', 'Chelsea'),
        ('Manchester United', 'Tottenham'),
        ('Liverpool', 'Manchester City'),
        ('Brighton', 'Newcastle'),
        ('West Ham', 'Crystal Palace')
    ]
    
    print("\n--- GAMEWEEK PREDICTIONS ---")
    predictions = predictor.predict_gameweek(fixtures)
    
    # Backtest performance
    print("\n--- BACKTESTING ---")
    accuracy = predictor.backtest_predictions(test_matches=50)
    
    # Save model for later use
    predictor.save_model('example_model.pkl')
    
    return predictor

def model_persistence_example():
    """Example of saving and loading models"""
    print("\n=== MODEL PERSISTENCE EXAMPLE ===")
    
    # Train a model
    predictor1 = ImprovedPLPredictor()
    
    if predictor1.get_football_data():
        if predictor1.train_models():
            # Save the model
            predictor1.save_model('saved_predictor.pkl')
            
            # Create a new predictor and load the saved model
            predictor2 = ImprovedPLPredictor()
            
            # Load match data (needed for predictions)
            predictor2.get_football_data()
            
            # Load the trained model
            if predictor2.load_model('saved_predictor.pkl'):
                print("Model loaded successfully!")
                
                # Use the loaded model
                predictor2.predict_match('Arsenal', 'Tottenham')

def batch_prediction_example():
    """Example of batch predictions with analysis"""
    print("\n=== BATCH PREDICTION EXAMPLE ===")
    
    predictor = ImprovedPLPredictor()
    
    if not predictor.get_football_data():
        return
    
    if not predictor.train_models():
        return
    
    # Define a full gameweek
    gameweek_fixtures = [
        ('Arsenal', 'Aston Villa'),
        ('Brighton', 'Brentford'),
        ('Chelsea', 'Crystal Palace'),
        ('Everton', 'Fulham'),
        ('Leicester', 'Liverpool'),
        ('Manchester City', 'Newcastle'),
        ('Nottingham Forest', 'Sheffield United'),
        ('Tottenham', 'West Ham'),
        ('Bournemouth', 'Manchester United'),
        ('Wolves', 'Burnley')
    ]
    
    print(f"Predicting {len(gameweek_fixtures)} matches...")
    predictions = predictor.predict_gameweek(gameweek_fixtures)
    
    # Analyze predictions
    home_wins = sum(1 for p in predictions if p['prediction'] == 'H')
    draws = sum(1 for p in predictions if p['prediction'] == 'D')
    away_wins = sum(1 for p in predictions if p['prediction'] == 'A')
    
    print(f"\nPrediction Summary:")
    print(f"Home wins: {home_wins}")
    print(f"Draws: {draws}")
    print(f"Away wins: {away_wins}")
    
    # High confidence predictions
    high_confidence = [p for p in predictions if p['confidence'] > 0.6]
    print(f"\nHigh confidence predictions ({len(high_confidence)}):")
    for pred in high_confidence:
        print(f"  {pred['fixture']}: {pred['prediction']} ({pred['confidence']:.1%})")

def custom_team_analysis():
    """Analyze specific teams in detail"""
    print("\n=== CUSTOM TEAM ANALYSIS ===")
    
    predictor = ImprovedPLPredictor()
    
    if not predictor.get_football_data():
        return
    
    if not predictor.train_models():
        return
    
    # Analyze a specific team against various opponents
    focus_team = 'Liverpool'
    opponents = ['Manchester City', 'Arsenal', 'Chelsea', 'Manchester United', 'Tottenham']
    
    print(f"Analyzing {focus_team} against top opponents:")
    
    home_results = []
    away_results = []
    
    for opponent in opponents:
        # Home fixture
        pred_h, prob_h = predictor.predict_match(focus_team, opponent, show_details=False)
        home_results.append({
            'opponent': opponent,
            'prediction': pred_h,
            'win_prob': prob_h[list(predictor.best_model.classes_).index('H')]
        })
        
        # Away fixture
        pred_a, prob_a = predictor.predict_match(opponent, focus_team, show_details=False)
        away_results.append({
            'opponent': opponent,
            'prediction': pred_a,
            'win_prob': prob_a[list(predictor.best_model.classes_).index('A')]
        })
    
    # Summary
    home_df = pd.DataFrame(home_results)
    away_df = pd.DataFrame(away_results)
    
    print(f"\n{focus_team} Home Fixtures:")
    print(home_df.to_string(index=False))
    
    print(f"\n{focus_team} Away Fixtures:")
    print(away_df.to_string(index=False))

if __name__ == "__main__":
    print("Premier League Predictor - Example Usage")
    print("=" * 50)
    
    # Run examples
    try:
        predictor = basic_example()
        advanced_example()
        batch_prediction_example()
        custom_team_analysis()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have an internet connection for data download.")


```
