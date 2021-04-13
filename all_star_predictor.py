import sys
import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn import svm
import time

era = int(sys.argv[1])

YX = pandas.read_csv( "data/season_stats_AS.csv" )

if era == 1:
    feature_list = ['Age', 'G', 'TS%', 'FTr', 'WS', 'FGA', 'FG%', 'eFG%', 'FTA', 'FT%', 'TRB', 'AST', 'PTS']
elif era == 2:
    feature_list = ['Age', 'G', 'TS%', 'FTr', 'WS', 'FGA', 'FG%', 'eFG%', 'FTA', 'FT%', 'TRB', 'AST', 'PTS', 'MP', 'PER', 'STL', 'BLK']
    YX = YX[YX['Year']>=1974]
elif era == 3:
    feature_list = ['Age', 'G', 'TS%', 'FTr', 'WS', 'FGA', 'FG%', 'eFG%', 'FTA', 'FT%', 'TRB', 'AST', 'PTS', 'MP', 'PER', 'STL', 'BLK', 'USG%', '3PA', '3P%']
    YX = YX[YX['Year']>=1980]
    
YX.dropna(subset = feature_list, inplace=True)


Y = YX['AS'].to_numpy()

#unique_vals = list(set(Y))

#for i, val in enumerate(unique_vals):
	#Y[ Y == val ] = i

Y = Y.astype('int')

X = YX[feature_list]

print(Y)
print(X)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.3, random_state = 42)

model = svm.SVC()

start = time.time()

model.fit(Xtrain.values, Ytrain)

end = time.time()

correct = 0
false_pos = 0
false_neg = 0
for i, x in enumerate(Xtest.values):
    yhat = model.predict(x.reshape(1, -1))
    if Ytest[i] == yhat:
        correct += 1
    else:
        if yhat == 0:
            false_neg+=1
        else:
            false_pos+=1
        
false_pos_tot = 0
false_neg_tot = 0
correct_tot = 0
for i, player in YX.iterrows():
    x = player[feature_list]
    prediction = model.predict(x.to_numpy().reshape(1, -1))
    if prediction == 1:
        print(player['Player'], player['Year'])
        print('True Label: {}', player['AS'])
        if int(player['AS']) == 0:
            false_pos_tot += 1
        else:
            correct_tot += 1
    else:
        if int(player['AS']) == 1:
            false_neg_tot += 1
        else:
            correct_tot += 1
        

print("\nTest Accuracy: {} out of {} ({})".format(correct, len(Xtest), correct/len(Xtest)))
print("False Positives on Test Set: {}".format(false_pos))
print("False Negatives on Test Set: {}".format(false_neg))
print("\nComplete Accuracy: {} out of {} ({})".format(correct_tot, len(YX), correct_tot/len(YX)))
print("False Positives on Complete Set: {}".format(false_pos_tot))
print("False Negatives on Complete Set: {}".format(false_neg_tot))
print("\nTime elapsed: {} seconds".format(end - start))
