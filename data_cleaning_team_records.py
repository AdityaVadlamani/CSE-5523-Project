import sys
import numpy
import pandas


YX = pandas.read_csv( "data/Team_Records.csv" )

for index, row in YX.iterrows():
    
    if row['Team'][-1] == '*':
        YX.loc[index,'Team'] = row['Team'][:-1]
    
    if row['Season'] == '1999-00':
        YX.loc[index,'Season'] = '2000'
    else:
        YX.loc[index,'Season'] = row['Season'][ : 2] + row['Season'][-2 : ]

YX.to_csv('data/Team_Records_Cleaned.csv', index = False)
