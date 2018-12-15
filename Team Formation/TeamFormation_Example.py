# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import math


# In[86]:

toRun = pd.read_csv('mergedDataset.csv')

#print(toRun.head(0))
toRun = toRun[['id', 'player_position', 'player_name', 'last10_ratio_cleanSheets_opp', 'last10_ratio_cleanSheets_own', 'last10_ratio_wins_opp', 'last10_ratio_wins_own', 'last3_assists', 'last3_goals', 'last3_ratio_points', 'last3_ycards', 'opp_team_rank', 'player_team_rank', 'player_value', 'ratio_assists', 'ratio_attempted_passes', 'ratio_big_chancesCreated', 'ratio_big_chancesMiss', 'ratio_creativity', 'ratio_dribbles', 'ratio_fouls', 'ratio_goals_conceded_opp_team', 'ratio_goals_conceded_player_team', 'ratio_goals_opp_team', 'ratio_goals_player_team', 'ratio_goals_scored', 'ratio_key_passes', 'ratio_leading_goal', 'ratio_minutes_played', 'ratio_offsides', 'ratio_open_playcross', 'ratio_own_goals', 'ratio_penalties_conceded', 'ratio_penalties_missed', 'ratio_penalties_saved', 'ratio_saves', 'ratio_selection', 'ratio_tackles', 'ratio_threat', 'week_no', 'season', 'week_points']]


# In[87]:

testSet = toRun.loc[((toRun['season'] == 2019) & (toRun['week_no'] == 7))]
X_test = testSet.loc[:,[i for i in list(testSet.columns) if i not in ['week_points','player_position','id', 'player_name', 'player_team', 'player_class', 'season']]]
Y_test = pd.DataFrame(testSet.loc[:, testSet.columns == 'player_class'])


# In[89]:

############
y_pred = rf.predict(X_test)
#print(testSet.head())
counter = 0
for i, counter in zip(y_pred, range(len(testSet))):
    #print(y_pred[counter])
    if(y_pred[counter] == 'A'):
        #print(testSet.iloc[counter,2])
        print(testSet.iloc[counter,41])


# In[49]:

trainingSet = toRun.loc[((toRun['season'] == 2019) & (toRun['week_no'] != 8))|(toRun['season'] == 2018)]
X_train = trainingSet.loc[:,[i for i in list(trainingSet.columns) if i not in ['week_points','player_position','id', 'player_name', 'player_team']]]
Y_train = pd.DataFrame(trainingSet.loc[:, trainingSet.columns == 'week_points'])


# In[50]:

reg = linear_model.Lasso(alpha = 0.1)
reg.fit(X_train, Y_train)
y_pred = reg.predict(testSet.loc[:,[i for i in list(testSet.columns) if i not in ['week_points','player_position','id', 'player_name', 'player_team']]])

print('RMSE: ', math.sqrt(mean_squared_error(testSet['week_points'], y_pred)))


# In[51]:

finalSet = testSet.copy(deep=True)
finalSet['score'] = y_pred


# In[52]:

from itertools import permutations
import pandas as pd

#Check that we have maximum 3 players from each team
def checkTeamLimit(cntTeam):
    cntTeamSet = list(set(cntTeam))
    for i in cntTeamSet:
        counter = 0
        for j in cntTeam:
            if i == j:
                counter += 1
        if counter > 3:
            return False
    return True
#Get the best expected line-up for each formation
def getBestLineup(scoresDF, threshold = [3, 8, 8, 6], budget = 850):
    formations = [[4, 4, 2], [4, 3, 3], [3, 4, 3], [3, 5, 2]]
    scoresDF = scoresDF.sort_values(by='score', ascending=False)
    weekPointsDF = scoresDF.sort_values(by='week_points', ascending=False)
    
    pvInd = scoresDF.columns.get_loc("player_value")
    pnInd = scoresDF.columns.get_loc("player_name")
    wpInd = scoresDF.columns.get_loc("week_points")
    sInd = scoresDF.columns.get_loc("score")
    ptInd = scoresDF.columns.get_loc("player_team")
    
    #Update Current selected permutation of team lineup
    def updateCnt(i, df, cntPrice, cntFor, cntActual, cntScore, line, cntTeam, cntMax):
        cntPrice += df[i, pvInd]
        cntFor.append(df[i, pnInd])
        cntActual += df[i, wpInd]
        cntScore += df[i, sInd]
        cntMax = max(cntMax, df[i, sInd])
        cntTeam.append(df[i, ptInd])
        return cntPrice, cntFor, cntActual, cntScore, cntTeam, cntMax
    
    GKs = (scoresDF[scoresDF.player_position == 'GKP']).values
    FWs = (scoresDF[scoresDF.player_position == 'FWD']).values
    MDs = (scoresDF[scoresDF.player_position == 'MID']).values
    DFs = (scoresDF[scoresDF.player_position == 'DEF']).values
    
    #get Week Dream Team
    GKs2 = weekPointsDF[weekPointsDF.player_position == 'GKP']
    FWs2 = weekPointsDF[weekPointsDF.player_position == 'FWD']
    MDs2 = weekPointsDF[weekPointsDF.player_position == 'MID']
    DFs2 = weekPointsDF[weekPointsDF.player_position == 'DEF']
    DreamTeam = []
    for i in range(threshold[0]):
        DreamTeam.append(GKs2.iloc[i].player_name)
    for i in range(threshold[1]):
        DreamTeam.append(DFs2.iloc[i].player_name)
    for i in range(threshold[2]):
        DreamTeam.append(MDs2.iloc[i].player_name)
    for i in range(threshold[3]):
        DreamTeam.append(FWs2.iloc[i].player_name)
        
    #get the best line-up for each formation
    for formation in formations:
        maxi = 0 #Score of Best Line-up
        bestFor = 0 #Best Line-up
        bestPrice = 0 #Price of Best-Lineup
        bestActual = 0 #Actual Score of selected best Formation
        
        #All permutations of Goal Keepers
        gkStr = '1' + '0' * (threshold[0] - 1)
        gks = list(set([''.join(p) for p in permutations(gkStr)]))
        #All permutations of defenders
        dfStr = '1' * formation[0] + '0' * (threshold[1] - formation[0])
        dfs = list(set([''.join(p) for p in permutations(dfStr)]))
        #All permutations of midfielders
        mdStr = '1' * formation[1] + '0' * (threshold[2] - formation[1])
        mds = list(set([''.join(p) for p in permutations(mdStr)]))
        #All permutations of Forwards
        fwStr = '1' * formation[2] + '0' * (threshold[3] - formation[2])
        fws = list(set([''.join(p) for p in permutations(fwStr)]))
        
        #Try all permutations of players
        for gk in gks:
            for df in dfs:
                for md in mds:
                    for fw in fws:
                        #variables to store cnt permutation (Price - Expected Score - Lineup Names - Actual Score)
                        cntPrice = 0
                        cntScore = 0
                        cntFor = []
                        cntActual = 0
                        cntMax = 0 # to know maximum expected points lineup to make the highest one as the captain 
                        cntTeam = []
                        for i in range(len(gk)):
                            if gk[i] == '1':
                                cntPrice, cntFor, cntActual, cntScore, cntTeam, cntMax = updateCnt(i, GKs, cntPrice, cntFor, cntActual, cntScore, 'GKP', cntTeam, cntMax)
                        for i in range(len(df)):
                            if df[i] == '1':
                                cntPrice, cntFor, cntActual, cntScore, cntTeam, cntMax = updateCnt(i, DFs, cntPrice, cntFor, cntActual, cntScore, 'DEF', cntTeam, cntMax)
                        for i in range(len(md)):
                            if md[i] == '1':
                                cntPrice, cntFor, cntActual, cntScore, cntTeam, cntMax = updateCnt(i, MDs, cntPrice, cntFor, cntActual, cntScore, 'MID', cntTeam, cntMax)
                        for i in range(len(fw)):
                            if fw[i] == '1':
                                cntPrice, cntFor, cntActual, cntScore, cntTeam, cntMax = updateCnt(i, FWs, cntPrice, cntFor, cntActual, cntScore, 'FWD', cntTeam, cntMax)
                        
                        cntScore = cntMax + cntScore # Captain Score is doubled
                        if cntPrice <= budget and cntScore > maxi and checkTeamLimit(cntTeam): #Check Budget - Maximum Score - 3 players max from each team 
                            maxi = cntScore
                            bestFor = cntFor
                            bestPrice = cntPrice
                            bestActual = cntActual
        counter = 0
        for p in bestFor:
            if p in DreamTeam:
                counter += 1
        bestPercent = counter / 11.0 * 100
        print('###################### Formation:', formation, '##########################\n', 'Line-Up: ', bestFor)
        print('Price: ', bestPrice, '\nExpected Score: ', maxi, '\nActual Score:', bestActual, '\nPercent from DreamTeam:', bestPercent)


# In[53]:

getBestLineup(finalSet)
