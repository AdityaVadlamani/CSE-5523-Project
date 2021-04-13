import sys
import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn import svm
import time


YX = pandas.read_csv( "data/Seasons_stats.csv" )
YX = YX.drop_duplicates(subset=['Year', 'Player'], keep="first", ignore_index = True)
YX.dropna(subset = ['Pos', 'PTS', 'AST', 'FT%', '2P%', 'FG%', 'WS', 'TS%'], inplace=True)
for index, row in YX.iterrows():
    
    if row['Player'][-1] == '*':
        YX.loc[index,'Player'] = row['Player'][:-1]
    
    if row['Player'] == 'Metta World':
        print(row['Player'])
        YX.loc[index,'Player'] = 'Metta World Peace'

YX.to_csv('data/season_stats_cleaned.csv', index = False)

