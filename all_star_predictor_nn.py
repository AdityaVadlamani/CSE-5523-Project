import sys
import numpy
import pandas
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.utils import resample, shuffle
import time

EPSILON = 1E-7
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

YX_pos = YX[YX.AS == 1]
YX_neg = YX[YX.AS == 0]

YX_pos_upsampled = resample(YX_pos, replace=True, n_samples = len(YX_neg), random_state = 0)

YX_upsampled = shuffle(pandas.concat([YX_neg, YX_pos_upsampled], ignore_index=True), random_state = 42)

Y = YX_upsampled['AS'].to_numpy().astype('int')

X = YX_upsampled[feature_list]

#X['AS'] = Y.tolist()

#XTrue, YTrue = X[X['AS'] == 1], Y[Y == 1]
#XFalse, YFalse = X[X['AS'] == 0], Y[Y == 0]

#X = X.drop(['AS'], axis=1)
#XTrue = XTrue.drop(['AS'], axis=1)
#XFalse = XFalse.drop(['AS'], axis=1)

for trial in range(10): 

    model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = X.columns.shape),
    tf.keras.layers.Dense((len(feature_list) + 2) // 2, activation ='relu'),
    tf.keras.layers.Dense((len(feature_list) + 2) // 2, activation ='relu'),
    tf.keras.layers.Dense(2, activation = 'softmax')
    ])

    model.compile(optimizer='adam',
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['sparse_categorical_accuracy'])

    #XTrueTrain, XTrueTest, YTrueTrain, YTrueTest = train_test_split(XTrue, YTrue, test_size = 0.3, random_state = trial)
    #XFalseTrain, XFalseTest, YFalseTrain, YFalseTest = train_test_split(XFalse, YFalse, test_size = 0.3, random_state = trial)

    #Xtrain, Xtest, Ytrain, Ytest = XTrueTrain.append(XFalseTrain, ignore_index = True), XTrueTest.append(XFalseTest, ignore_index = True),\
#numpy.append(YTrueTrain, YFalseTrain), numpy.append(YTrueTest, YFalseTest)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = .10, random_state = trial)
    Xtrain, Xval, Ytrain, Yval  = train_test_split(Xtrain, Ytrain, test_size= 1 / 9, random_state=trial)

    start = time.time()

    model.fit(Xtrain, Ytrain, epochs = 15, verbose = 0, class_weight={1 : .75, 0: .25}, validation_data=(Xval, Yval))

    end = time.time()

    predictions = model.predict(Xtest)

    correct = 0
    total_pos = numpy.sum(Ytest)
    total_neg = len(Ytest) - total_pos
    
    false_pos = 0
    false_neg = 0
    true_pos = 0
    true_neg = 0
    for i, _ in enumerate(predictions):
        yhat = numpy.argmax(predictions[i])
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

    precision = (true_pos)/(true_pos + false_pos + EPSILON)
    recall = (true_pos)/(true_pos + false_neg + EPSILON)
    F1 = 2 * (precision * recall)/(precision + recall + EPSILON)

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

    print("Test Precision: {}/{} (= {})".format(true_pos, true_pos + false_pos, precision))
    print("Test Recall: {}/{} (= {})".format(true_pos, true_pos + false_neg, recall))
    print("Test F1 Score: {}".format(F1))


    #print("\nComplete Accuracy: {} out of {} ({})".format(correct_tot, len(YX), correct_tot/len(YX)))
    #print("False Positives on Complete Set: {}".format(false_pos_tot))
    #print("False Negatives on Complete Set: {}".format(false_neg_tot))
    print("\nTime elapsed: {} seconds".format(end - start))
    print()
