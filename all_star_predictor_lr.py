import sys
import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample, shuffle
import time
EPSILON = 1E-7

era = int(sys.argv[1])

YX = pandas.read_csv( "data/season_stats_AS_records_norm_2.csv" )

if era == 1:
    feature_list = ['Age', 'G', 'TS%', 'FTr', 'WS', 'FG%', 'eFG%', 'FTA', 'FT%', 'TRB', 'AST', 'PTS', 'W/L']
elif era == 2:
    feature_list = ['Age', 'G', 'TS%', 'FTr', 'WS', 'FG%', 'eFG%', 'FTA', 'FT%', 'TRB', 'AST', 'PTS', 'MP', 'PER', 'STL', 'BLK', 'W/L']
    YX = YX[YX['Year']>=1974]
elif era == 3:
    feature_list = ['Age', 'G', 'TS%', 'FTr', 'WS', 'FG%', 'eFG%', 'FTA', 'FT%', 'TRB', 'AST', 'PTS', 'MP', 'PER', 'STL', 'BLK', 'USG%', '3PA', '3P%', 'W/L']
    YX = YX[YX['Year']>=1980]
    
YX.dropna(subset = feature_list, inplace=True)
YX.reset_index(drop=True, inplace=True)

YX_pos = YX[YX.AS == 1]
YX_neg = YX[YX.AS == 0]

YX_pos_upsampled = resample(YX_pos, replace=True, n_samples = len(YX_neg), random_state = 0)

YX_upsampled = shuffle(pandas.concat([YX_neg, YX_pos_upsampled], ignore_index=True), random_state = 42)

Y = YX_upsampled['AS'].to_numpy().astype('int')

X = YX_upsampled[feature_list]

accs = []
fprs = []
fnrs = []
times = []

for trial in range(10):

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.3, random_state = trial)

    model = LogisticRegression(max_iter = 10000)

    start = time.time()

    model.fit(Xtrain.values, Ytrain)

    end = time.time()

    correct = 0
    false_pos = 0
    false_neg = 0
    true_pos = 0
    true_neg = 0
    
    for i, x in enumerate(Xtest.values):
        yhat = model.predict(x.reshape(1, -1))
        if Ytest[i] == yhat:
            correct += 1
            if yhat == 0:
                true_neg += 1
            else:
                true_pos += 1
        else:
            if yhat == 0:
                false_neg += 1
            else:
                false_pos += 1
                
    accs.append(correct/len(Xtest))
    fprs.append(false_pos / (false_pos + true_neg + EPSILON))
    fnrs.append(false_neg / (false_neg + true_pos + EPSILON))
    times.append(end - start)

    print("\nTrial: {}".format(trial + 1))
    print("Test Accuracy: {} out of {} ({})".format(correct, len(Xtest), correct/len(Xtest)))
    print("False Positive rate on Test Set: {}/{} (= {})".format(false_pos, false_pos + true_neg, false_pos / (false_pos + true_neg + EPSILON)))
    print("False Negative rate on Test Set: {}/{} (= {})".format(false_neg, false_neg + true_pos, false_neg / (false_neg + true_pos + EPSILON)))

    print("\nTime elapsed: {} seconds".format(end - start))
    print()

print("\nAverages across 10 trials:")
print("Test Accuracy: {}".format(sum(accs)/len(accs)))
print("False Positive rate on Test Set: {}".format(sum(fprs)/len(fprs)))
print("False Negative rate on Test Set: {}".format(sum(fnrs)/len(fnrs)))
print("\nTime elapsed: {} seconds".format(sum(times)/len(times)))


