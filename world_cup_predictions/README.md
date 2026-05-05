# World Cup Prediction CLI

A simple Python command-line tool for predicting World Cup match outcomes using logistic regression.

## Features

- Load match results from a CSV file
- Train a logistic regression model
- Predict the home team's win probability for a future match
- Split data into train/test sets

## Setup

1. Install Python 3.10+.
2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Split the default dataset into train and test sets:

```bash
python worldcup_predict.py --split-data
```

Predict the home team's win probability:

```bash
python worldcup_predict.py --predict England Iran
```

## CSV format

The matches file must include these columns:

- `date`
- `home_team`
- `away_team`
- `home_score`
- `away_score`

