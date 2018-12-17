# List of the extracted features from Raw Data of Kaggle Datasets / Fantasy Premier League API:
## Each Instance represent Player Info before a specific Game Week (Week_Points is the target column which is required to be predicted)
### The player info (features) in each instance includes all the following:
#### Player History:
  1. Player Position (Forward, Midfielder, Defender, Goalkeeper)
  2. Ratio of Goals Scored/ week in season for player
  3. Player Goals scored last 3 weeks
  4. Player Yellow Cards last 3 weeks 
  5. Ratio of fouls/week made by the player 
  6. Ratio of Goals Scored/ week in season for player
  7. Ratio of Assists/ week in season for player
  8. Ratio of player points/week in last 3 weeks
  9. Ratio of player points/week in last season 
  10. Ratio of minutes played/week in current season 
  11. Ratio of saves/week for the player in the current season (GK)
  12. Ratio of clean sheets for player own team in the last 10 matches
  13. Ratio of fouls/week made by player
  14. Ratio of tackles won/ week made by player

#### Player Own Team History:
  15. Player Team Rank
  16. Ratio of Goals scored by team/week in season for own team
  17. Ratio of Goals conceded by team / week in season for own team
  18. Ratio of wins/losses for player own team in the last 10 matches

#### Opponent Team History:
  19. Opponent Team Rank
  20. Ratio of Goals scored by team/week in season for the opponent team
  21. Ratio of Goals conceded by team / week in season for the opponent team
  22. Ratio of clean sheets for the opponent team in the last 10 matches
  23. Ratio of wins/losses for the opponent team in the last 10 matches
