import pandas as pd 
import datetime
import json
import numpy as np

season = 2017 #Season Year to extract the data from
season_players = pd.DataFrame(list(db.old_fantasy_players.find({'season':int(season)})))
playersids= {2017: pd.read_csv('../Data/Raw Data/Season 2017/player_ids.csv'),
             2018: pd.read_csv('../Data/Raw Data/Season 2018/player_ids.csv'),
			 2019: pd.read_csv('../Data/Raw Data/Season 2019/player_ids.csv')}

teams_dataset = pd.read_csv('../Data/Raw Data/teams_standings_' + str(season) + '.csv')
season_ids = playersids[int(season)]
season_ids['player_name'] = season_ids.first_name + ' ' + season_ids.second_name

players_raw = pd.read_csv('../Data/Raw Data/Season ' + season + '/players_raw_data.csv')
alphabet_teams = sorted(teams_dataset.team_name.unique())

listteam_code = list(range(1,21))
team_map_code={}
for i in range(len(alphabet_teams)):
    team_map_code[listteam_code[i]]=alphabet_teams[i]
    
    
season_players['player_team_name']= " "
season_players['player_position']=" "
season_players['player_team_id']=0.0
for player_id in tq(season_players.element.unique()):
    team_code= players_raw[players_raw.id==player_id].team.values[0]
    player_type= players_raw[players_raw.id==player_id].element_type.values[0]
    
    if player_type==1:
        player_pos='GKP'
    elif player_type==2:
        player_pos='DEF'
    elif player_type==3:        
        player_pos='MID'
    elif player_type==4:
        player_pos='FWD'
        
    season_players.loc[season_players.element==player_id , 'player_team_name']=team_map_code[team_code]
    season_players.loc[season_players.element==player_id , 'player_team_id']=team_code
    season_players.loc[season_players.element==player_id , 'player_position']=player_pos
    

all_players_seasons=[]

for player_id in tq(season_ids.id.unique()):
    player_weeks= season_players[season_players.element==player_id]
    player_weeks= player_weeks.sort_values('gw_no')
    full_name= player_weeks.name.values[0].split('_')[:2]
    full_name= "_".join(full_name)
    
    player_team= player_weeks.player_team_name
    if player_team.shape[0]!=0:
        player_team= teams_dataset[teams_dataset.team_name==player_team.values[0]].sort_values('week_no')

        for week in player_weeks.gw_no.unique():
            opp_team_name=team_map_code[player_weeks[player_weeks.gw_no==week].opponent_team.values[0]]
            opp_team= teams_dataset[teams_dataset.team_name==opp_team_name]
            last3_data= player_weeks.iloc[max(0,week-4):week].iloc[:-1]
            last10_data= player_weeks.iloc[max(0,week-10):week].iloc[:-1]


            if last3_data.shape[0]>0:
                player_instance={'id': player_id, 'player_team_name':player_team.team_name.values[0] ,
                                 'week_no':week, 'player_name':full_name , 
                                 'player_position': player_weeks.player_position.values[0],
                                'week_points': season_players[(season_players.element==player_id) 
                                                              & (season_players.gw_no==week)].total_points.values[0]}
                all_last= player_weeks[:week-1]

                player_instance['last3_goals']= last3_data.goals_scored.sum()
                player_instance['last3_ycards']=  last3_data.yellow_cards.sum()
                player_instance['last3_assists']= last3_data.assists.sum()
                player_instance['player_opp_name']= opp_team.team_name.values[0]
                player_instance['player_team_rank']= teams_dataset[(teams_dataset.team_name==player_instance['player_team_name']) & (teams_dataset.week_no==week)].ranking.values[0]
                player_instance['opp_team_rank']= teams_dataset[(teams_dataset.team_name==player_instance['player_opp_name']) & (teams_dataset.week_no==week)].ranking.values[0]
                player_instance['ratio_goals_scored']= all_last.goals_scored.mean() 
                player_instance['ratio_assists']= all_last.assists.mean()
                player_instance['last3_ratio_points']= last3_data.total_points.sum()/min(last3_data.shape[0],3.0)
                player_instance['ratio_minutes_played']= all_last.minutes.mean()
                player_instance['ratio_saves']= all_last.saves.mean()

                last10_teamData= player_team.iloc[max(0,week-10):week].iloc[:-1]
                last10_oppteam= opp_team.iloc[max(0,week-10):week].iloc[:-1]
                all_teamData= player_team[:week-1]
                all_oppData= opp_team[:week-1]


                player_instance['ratio_goals_player_team']= all_teamData.goals_for.mean()
                player_instance['ratio_goals_opp_team']= all_oppData.goals_for.mean()
                player_instance['ratio_goals_conceded_player_team']= all_teamData.goals_against.mean()
                player_instance['ratio_goals_conceded_opp_team']= all_teamData.goals_against.mean()
                player_instance['last10_ratio_cleanSheets_own']= last10_teamData.clean_sheets.sum()/min(last10_teamData.shape[0],10.0)
                player_instance['last10_ratio_cleanSheets_opp']= last10_oppteam.clean_sheets.sum()/min(last10_oppteam.shape[0],10.0)
                player_instance['last10_ratio_wins_own']= last10_teamData.wins.sum()/min(last10_teamData.shape[0],10.0)
                player_instance['last10_ratio_wins_opp']= last10_oppteam.wins.sum()/min(last10_oppteam.shape[0],10.0)
                player_instance['ratio_fouls']= all_last.fouls.mean()
                player_instance['ratio_tackles']= all_last.tackles.mean()
                player_instance['player_value']= player_weeks[player_weeks.gw_no==week].value.values[0]
                player_instance['ratio_key_passes']= all_last.key_passes.mean()
                player_instance['ratio_attempted_passes']= all_last.attempted_passes.mean()
                player_instance['ratio_big_chancesCreated']= all_last.big_chances_created.mean()
                player_instance['ratio_big_chancesMiss']= all_last.big_chances_missed.mean()
                player_instance['ratio_dribbles']= all_last.dribbles.mean()
                player_instance['ratio_leading_goal']= all_last.errors_leading_to_goal.mean() 
                player_instance['ratio_own_goals']= all_last.own_goals.mean()
                player_instance['ratio_open_playcross']= all_last.open_play_crosses.mean()
                player_instance['ratio_offsides']= all_last.offside.mean()
                player_instance['ratio_penalties_missed']= all_last.penalties_missed.mean() 
                player_instance['ratio_penalties_saved']= all_last.penalties_saved.mean()
                player_instance['ratio_penalties_conceded']= all_last.penalties_conceded.mean()
                player_instance['ratio_selection']= all_last.selected.mean()
                player_instance['ratio_creativity']= all_last.creativity.mean()
                player_instance['ratio_threat']= all_last.threat.mean()
                all_players_seasons.append(player_instance)
                
all_players_seasons= pd.DataFrame(all_players_seasons)
all_players_seasons.to_csv('../Data/Extracted Data/dataset_extracted_'+str(season)+'.csv')
