import sys
import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn import svm
import time

all_stars = pandas.read_csv( "data/nba_all_stars.csv" )

as_dict = {}

for index, row in all_stars.iterrows():
    
    as_dict[row['Player']] = []
    
    year_string = row['Selections']
    
    year_splits = year_string.split('; ')
    #print(row['Player'])
    #print(year_splits)
    
    for element in year_splits:
        if '–' in element:
            bounds = element.split('–')
            #print(bounds)
            for i in range(int(bounds[0]), int(bounds[1]) + 1):
                #print(i)
                as_dict[row['Player']].append(str(i))
        else:
            as_dict[row['Player']].append(element)

print(as_dict)

YX = pandas.read_csv( "data/season_stats_cleaned.csv" )

as_labels = []

for index, row in YX.iterrows():
    if row['Player'] in as_dict and str(int(row['Year'])) in as_dict[row['Player']]:
        print(row['Player'], row['Year'])
        as_labels.append(1)
    else:
        as_labels.append(0)

YX['AS'] = as_labels

YX = YX[YX['Year'] != 1950]
YX = YX[YX['Year'] != 1999]
YX.to_csv('data/season_stats_AS.csv', index=False)
