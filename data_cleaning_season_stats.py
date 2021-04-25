import sys
import numpy
import pandas

YX = pandas.read_csv( "data/Seasons_stats.csv" )
YX2 = pandas.read_csv( "data/Seasons_stats.csv" )
YX2 = YX2.sort_values(by = ['G'], ascending=False)
YX2 = YX2[ YX2.Tm != 'TOT']

YX = YX.drop_duplicates(subset=['Year', 'Player'], keep="first", ignore_index = True)
YX2 = YX2.drop_duplicates(subset=['Year', 'Player'], keep="first", ignore_index = True)

YX2[['First','Last']] = YX2.Player.str.split(expand=True)

YX2 = YX2.sort_values(by =['Year', 'Last'], ascending=True)
YX2 = YX2.drop(['First', 'Last'], axis = 1)

YX.dropna(subset = ['Pos', 'PTS', 'AST', 'FT%', '2P%', 'FG%', 'WS', 'TS%'], inplace=True)

YX.reset_index(drop = True, inplace=True)
YX2.reset_index(drop = True, inplace=True)

for index, row in YX.iterrows():
	if row['Tm'] == 'TOT':
		idx = YX2.index[(row['Player'] == YX2['Player']) & (row['Year'] == YX2['Year'])].tolist()
		YX.loc[index,'Tm'] = YX2.loc[idx[0], 'Tm']

	if row['Player'][-1] == '*':
		YX.loc[index,'Player'] = row['Player'][:-1]

	if row['Player'] == 'Metta World':
		YX.loc[index,'Player'] = 'Metta World Peace'

YX.to_csv('data/season_stats_cleaned.csv', index = False)

