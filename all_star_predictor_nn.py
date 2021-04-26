import sys
import numpy
import pandas
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time

tf.random.set_seed(42)

era = int(sys.argv[1])

YX = pandas.read_csv( "data/season_stats_AS_records.csv" )

if era == 1:
    feature_list = ['Age', 'G', 'TS%', 'FTr', 'WS', 'FGA', 'FG%', 'eFG%', 'FTA', 'FT%', 'TRB', 'AST', 'PTS']
elif era == 2:
    feature_list = ['Age', 'G', 'TS%', 'FTr', 'WS', 'FGA', 'FG%', 'eFG%', 'FTA', 'FT%', 'TRB', 'AST', 'PTS', 'MP', 'PER', 'STL', 'BLK']
    YX = YX[YX['Year']>=1974]
elif era == 3:
    feature_list = ['Age', 'G', 'TS%', 'FTr', 'WS', 'FGA', 'FG%', 'eFG%', 'FTA', 'FT%', 'TRB', 'AST', 'PTS', 'MP', 'PER', 'STL', 'BLK', 'USG%', '3PA', '3P%']
    YX = YX[YX['Year']>=1980]
    
YX.dropna(subset = feature_list, inplace=True)
YX.reset_index(drop=True, inplace=True)

Y = YX['AS'].to_numpy().astype('int')

X = YX[feature_list]

X['AS'] = Y.tolist()

XTrue, YTrue = X[X['AS'] == 1], Y[Y == 1]
XFalse, YFalse = X[X['AS'] == 0], Y[Y == 0]

X.drop(['AS'], axis=1)

for trial in range(10): 

    model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = X.columns.shape),
    tf.keras.layers.Dense((len(feature_list) + 2) // 2, activation='relu'),
    tf.keras.layers.Dense(2)
    ])

    model.compile(optimizer='adam',
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy'])


    XTrueTrain, XTrueTest, YTrueTrain, YTrueTest = train_test_split(XTrue, YTrue, test_size = 0.3, random_state = trial)
    XFalseTrain, XFalseTest, YFalseTrain, YFalseTest = train_test_split(XFalse, YFalse, test_size = 0.3, random_state = trial)

    Xtrain, Xtest, Ytrain, Ytest = XTrueTrain.append(XFalseTrain, ignore_index = True), XTrueTest.append(XFalseTest, ignore_index = True),\
numpy.append(YTrueTrain, YFalseTrain), numpy.append(YTrueTest, YFalseTest)

    start = time.time()

    model.fit(Xtrain, Ytrain, epochs=10, verbose = 0)

    end = time.time()

    predictions = model.predict(Xtest)

    correct = 0
    false_pos = 0
    false_neg = 0
    total_pos = numpy.sum(Ytest)
    total_neg = len(Ytest) - total_pos
    for i, _ in enumerate(predictions):
        yhat = numpy.argmax(predictions[i])
        if Ytest[i] == yhat:
            correct += 1
        else:
            if yhat == 0:
                false_neg+=1
            else:
                false_pos+=1

    precision = (total_pos - false_pos)/total_pos
    recall = (total_pos - false_pos)/(total_pos - false_pos + false_neg)
    F1 = 2 * (precision * recall)/(precision + recall)
            
    #false_pos_tot = 0
    #false_neg_tot = 0
    #correct_tot = 0
    #
    #predictions = model.predict(X)
    #
    #for i, player in YX.iterrows():
    #    x = player[feature_list]
    #    prediction = numpy.argmax(predictions[i])
    #    if prediction == 1:
    #        #print(player['Player'], int(player['Year']))
    #        #print('True Label: {}'.format(player['AS']))
    #        if int(player['AS']) == 0:
    #            false_pos_tot += 1
    #        else:
    #            correct_tot += 1
    #    else:
    #        if int(player['AS']) == 1:
    #            false_neg_tot += 1
    #        else:
    #            correct_tot += 1
            
    print("\nTrial: {}".format(trial + 1))
    #print("Test Accuracy: {} out of {} ({})".format(correct, len(Xtest), correct/len(Xtest)))
    #print("False Positive rate on Test Set: {}/{} (= {})".format(false_pos, total_neg, false_pos / total_neg))
    #print("False Negative rate on Test Set: {}/{} (= {})".format(false_neg, total_pos, false_neg / total_pos))

    print("Test Precision: {}/{} (= {})".format(total_pos - false_pos, total_pos, precision))
    print("Test Recall: {}/{} (= {})".format(total_pos - false_pos, total_pos - false_pos + false_neg, recall))
    print("Test F1 Score: {}/{} (= {})".format(2 * precision * recall, precision + recall, F1))


    #print("\nComplete Accuracy: {} out of {} ({})".format(correct_tot, len(YX), correct_tot/len(YX)))
    #print("False Positives on Complete Set: {}".format(false_pos_tot))
    #print("False Negatives on Complete Set: {}".format(false_neg_tot))
    print("\nTime elapsed: {} seconds".format(end - start))
    print()
