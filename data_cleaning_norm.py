import sys
import numpy
import pandas

df = pandas.read_csv( "data/season_stats_AS_records_norm.csv" )

result = df.copy()

for feature_name in df.columns:
    if feature_name in ['Year', 'Player', 'Pos', 'Tm', 'Unnamed: 0', 'blanl', 'blank2', 'G']:
        result[feature_name] = df[feature_name]
    else:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

result.to_csv('data/season_stats_AS_records_norm_2.csv', index=False)
