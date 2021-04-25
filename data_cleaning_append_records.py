import sys
import numpy
import pandas

abbrevs = pandas.read_csv( "data/team_abbrevs.csv" )

abbrev_dict = {}
for index, row in abbrevs.iterrows():
    abbrev_dict[row['abbrev']]=row['team_name']

as_dict = {}

print(abbrev_dict)

YX = pandas.read_csv( "data/season_stats_AS.csv" )
records = pandas.read_csv( "data/Team_Records.csv" )

player_records = []
defunct_teams = ['INO', 'WSC', 'BLB']

defunct_records = {('BLB', 1951): 0.364,
                   ('BLB', 1952): 0.303,
                   ('BLB', 1953): 0.229,
                   ('BLB', 1954): 0.222,
                   ('BLB', 1955): 0.273,
                   ('INO', 1951): 0.456,
                   ('INO', 1952): 0.515,
                   ('INO', 1953): 0.394,
                   ('WSC', 1951): 0.286,
                  }

for index, row in YX.iterrows():
    if row['Tm'] in abbrev_dict:
        if row['Tm'] not in defunct_teams:
            season_row = records.loc[(records['Team'] == abbrev_dict[row['Tm']]) & (records['Season'] == row['Year'])]
            #print(type(season_row))
            player_records.append(season_row['W/L%'].item())
        else:
            player_records.append(defunct_records[(row['Tm'], row['Year'])])
    else:
        player_records.append('Manual Entry Required')

YX.insert(len(YX.columns) - 1, "W/L", player_records, True)

YX.to_csv('data/season_stats_AS_records.csv', index=False)
