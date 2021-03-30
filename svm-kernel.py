import sys
import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn import svm
import time

YX = pandas.read_csv( "data/Seasons_stats.csv" )
YX = YX.drop_duplicates(subset=['Year', 'Player'], keep="first", ignore_index = True)
YX.dropna(subset = ['Pos', 'PTS', 'AST', 'FT%', '2P%', 'FG%', 'WS', 'TS%'], inplace=True)

Y = YX['Pos'].to_numpy()

unique_vals = list(set(Y))

for i, val in enumerate(unique_vals):
	Y[ Y == val ] = i 

Y = Y.astype('int')

X = YX[['PTS', 'AST', 'FT%', '2P%', 'FG%', 'WS', 'TS%']]

print(Y)
print(X)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.3, random_state = 42)

model = svm.SVC()

start = time.time()

model.fit(Xtrain.values, Ytrain)

end = time.time()

correct = 0
for i, x in enumerate(Xtest.values):
	yhat = model.predict(x.reshape(1, -1))
	if Ytest[i] == yhat:
		correct += 1

print("Accuracy: {} out of {}".format(correct, len(Xtest)))
print("Time elapsed: {} seconds".format(end - start))