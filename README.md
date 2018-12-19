# Team Recommendation System for Fantasy Premier League:
  Fantasy Premier League (fantasy.premierleague.com) is also the most popular fantasy soccer game , with more than 5 million users, in the soccer world. Get your fantasy season started by drafting players to build a solid team and using coaching tips to keep your team going strong. 
  Every week, Fantasy Premier League users are competing to choose their team of players based on different criteria. The success of your fantasy football team hinges on your picks, and there is a different strategy to pick your draft. The main target is to minimize risk, maximize gain, and make the tough decisions when it comes to fantasy football draft day to chose the most useful team at the start of any league to get a high score. 

### The aim of this project is to:
  1. <b>Extract features</b> that can be used to predict player performance each game week from available datasets on Kaggle, and Fantasy API.
  2. Use different Machine Learning and Deep Learning algorithms to <b>build a model</b> that can predict player points each week.
Recommend a team line-up based on the results of the model that can achieves total score exceeding fantasy users scores consistently.

##### [Project Plan](https://goo.gl/GDquSU)
##### [Project Poster](https://github.com/mmaher22/FantasyPremierLeagueTeamRecommender/blob/master/ProjectPoster.pdf)


#### Available Data [URL](https://github.com/mmaher22/FantasyPremierLeagueTeamRecommender/tree/master/Data/Raw%20Data):
> <i>./Raw Data</i>

  We used data for seasons 2016\2017, 2017\2018, and current season 2018\2019 till the 8th week. As the current data is only considered as records for each game week, we need to make use of this data to find suitable features that can help in predicting the player performance in following week.
  
  
 #### Extracted Data [URL](https://github.com/mmaher22/FantasyPremierLeagueTeamRecommender/tree/master/Data/Extracted%20Data):
 > <i> ./Extracted Data</i>
 
 > <i> ./Feature Extraction/Players_Feature_Extractors.py</i>
 
 Each instance in the extracted dataset represents player information for a certain game week.
  This information includes:
      1)   Statistics of player performance in the past game weeks of the same season.
      2)   Team of the player past results and performance during last game weeks of same season.
      3)   Opponent Team of the player past results and performance during last game weeks of same season

 <img src="https://github.com/mmaher22/FantasyPremierLeagueTeamRecommender/blob/master/Feature%20Extraction/Picture2.png" width=300>
 <img src="https://github.com/mmaher22/FantasyPremierLeagueTeamRecommender/blob/master/Feature%20Extraction/Picture1.png" width=300)>
 
 
 #### Building Regression Model [URL](https://github.com/mmaher22/FantasyPremierLeagueTeamRecommender/tree/master/Model%20Building):
 > <i> ./Model Building</i>
 
 Different Regression Algorithms are tried out on the dataset with tuned parameters using Grid Search with CV = 5
 <img src = "https://github.com/mmaher22/FantasyPremierLeagueTeamRecommender/blob/master/Model%20Building/Picture3.png" width = 500>
 
 
 
#### Team Formation [URL](https://github.com/mmaher22/FantasyPremierLeagueTeamRecommender/tree/master/Team%20Formation)
> <i> ./Team Formation</i>

To Start forming a team-lineup based on the model predictions. We need to handle some constraints from Fantasy Game Rules:
  1. Team Budget should be less than 100 units. (We made our budget 85 units for 11 players without substitutions)
  2. We can’t choose more than 3 players from the same team.
  3. We should use one of the following formations (3-4-3 / 3-5-2 / 4-4-2 / 4-3-3 / 4-5-1).
  4. Score of Captain player selected is doubled.

Based on our analysis on dataset, average forwards players points are the highest, followed by midfielders, and defenders. So, we fixed 3-4-3 formation as our chosen formation for team-lineup selection but you can find results of all other formations here:
> <i> ./Team Formation/TeamFormation_Example.py</i>


#### Results:
Comparison between average scores by fantasy users and our model: <br>
<img src="https://github.com/mmaher22/FantasyPremierLeagueTeamRecommender/blob/master/Team%20Formation/Picture4.png" width=300>
<br>Examples of Chosen Team:<br>
<img src="https://github.com/mmaher22/FantasyPremierLeagueTeamRecommender/blob/master/Team%20Formation/Picture6.png?raw=true" width=200>
<img src="https://github.com/mmaher22/FantasyPremierLeagueTeamRecommender/blob/master/Team%20Formation/Picture7.png?raw=true" width=200>
