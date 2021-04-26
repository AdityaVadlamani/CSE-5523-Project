import sys
import numpy
import pandas
import matplotlib.pyplot as plt
import seaborn as sb

sb.set(rc={'figure.figsize':(12.5,8.5)})

feature_list_era_1 = ['Age', 'G', 'TS%', 'FTr', 'WS', 'FGA', 'FG%', 'eFG%', 'FTA', 'FT%', 'TRB', 'AST', 'PTS', 'W/L', 'AS']

feature_list_era_2 = ['Age', 'G', 'TS%', 'FTr', 'WS', 'FGA', 'FG%', 'eFG%', 'FTA', 'FT%', 'TRB', 'AST', 'PTS', 'MP', 'PER', 'STL', 'BLK', 'W/L', 'AS']

feature_list_era_3 = ['Age', 'G', 'TS%', 'FTr', 'WS', 'FGA', 'FG%', 'eFG%', 'FTA', 'FT%', 'TRB', 'AST', 'PTS', 'MP', 'PER', 'STL', 'BLK', 'USG%', '3PA', '3P%', 'W/L', 'AS']

YX = pandas.read_csv("data/season_stats_AS_records_norm.csv")

corr_matrix_1 = YX[feature_list_era_1].corr().round(2)

correlated_features_1 = corr_matrix_1.unstack()
correlated_features_1 = correlated_features_1[ correlated_features_1 > .7 ] 
print(correlated_features_1)

corr_matrix_2 = YX[YX['Year']>=1974]
corr_matrix_2 = corr_matrix_2[feature_list_era_2].corr().round(2)

correlated_features_2 = corr_matrix_2.unstack()
correlated_features_2 = correlated_features_1[ correlated_features_2 > .7 ] 
print(correlated_features_2)

corr_matrix_3 = YX[YX['Year']>=1980]
corr_matrix_3 = corr_matrix_3[feature_list_era_3].corr().round(2)

correlated_features_3 = corr_matrix_3.unstack()
correlated_features_3 = correlated_features_1[ correlated_features_3 > .7 ] 
print(correlated_features_3)

plt.figure()
sb.heatmap(data=corr_matrix_1, annot=True, center=0.0, cmap='coolwarm')

plt.figure()
sb.heatmap(data=corr_matrix_2, annot=True, center=0.0, cmap='coolwarm')

plt.figure()
sb.heatmap(data=corr_matrix_3, annot=True, center=0.0, cmap='coolwarm')
plt.show()
