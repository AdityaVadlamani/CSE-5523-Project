import sys
import numpy
import pandas


YX = pandas.read_csv( "data/season_stats_AS_records.csv" )

season_lengths = pandas.read_csv( "data/season_lengths.csv" )
"""
for index, row in season_lengths.iterrows():
    if row['Year'] == '1900':
        season_lengths.loc[index,'Year'] = '2000'
    else:
        season_lengths.loc[index,'Year'] = row['Year'][:2]+row['Year'][-2:]
        
season_lengths.to_csv('data/season_lengths.csv', index = False)

"""
season_lengths['Year'] = pandas.to_numeric(season_lengths['Year'])
for index, row in YX.iterrows():
    season_row = season_lengths.loc[season_lengths['Year'] == row['Year']]
    
    print(row['G'])
    print(season_row['Games'])
    
    YX.loc[index,'G'] = float(row['G']) / float(season_row['Games'].item())

YX.to_csv('data/season_stats_AS_records_norm.csv', index = False)
