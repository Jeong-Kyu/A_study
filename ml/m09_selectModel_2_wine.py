from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_wine
import warnings

warnings.filterwarnings('ignore')

dataset = load_wine()
x = dataset.data
y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=66)

allAlgorithm = all_estimators(type_filter = 'classifier')

for (name, algorithm) in allAlgorithm:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name,'의 정답률 :',accuracy_score(y_test,y_pred))
    except:
        print(name, '없는 것')

import sklearn
# print(sklearn.__version__) # 0.23.2

# AdaBoostClassifier 의 정답률 : 0.8888888888888888
# BaggingClassifier 의 정답률 : 0.9722222222222222
# BernoulliNB 의 정답률 : 0.4166666666666667
# CalibratedClassifierCV 의 정답률 : 0.9444444444444444
# CategoricalNB 없는 것
# CheckingClassifier 의 정답률 : 0.3888888888888889
# ClassifierChain 없는 것
# ComplementNB 의 정답률 : 0.6944444444444444
# DecisionTreeClassifier 의 정답률 : 0.9722222222222222
# DummyClassifier 의 정답률 : 0.4444444444444444
# ExtraTreeClassifier 의 정답률 : 0.9722222222222222
# ExtraTreesClassifier 의 정답률 : 1.0
# GaussianNB 의 정답률 : 1.0
# GaussianProcessClassifier 의 정답률 : 0.4444444444444444
# GradientBoostingClassifier 의 정답률 : 0.9722222222222222
# HistGradientBoostingClassifier 의 정답률 : 0.9722222222222222
# KNeighborsClassifier 의 정답률 : 0.6944444444444444
# LabelPropagation 의 정답률 : 0.5277777777777778
# LabelSpreading 의 정답률 : 0.5277777777777778
# LinearDiscriminantAnalysis 의 정답률 : 1.0
# LinearSVC 의 정답률 : 0.9166666666666666
# LogisticRegression 의 정답률 : 0.9722222222222222
# LogisticRegressionCV 의 정답률 : 0.9722222222222222
# MLPClassifier 의 정답률 : 0.5833333333333334
# MultiOutputClassifier 없는 것
# MultinomialNB 의 정답률 : 0.7777777777777778
# NearestCentroid 의 정답률 : 0.6944444444444444
# NuSVC 의 정답률 : 0.9444444444444444
# OneVsOneClassifier 없는 것
# OneVsRestClassifier 없는 것
# OutputCodeClassifier 없는 것
# PassiveAggressiveClassifier 의 정답률 : 0.6666666666666666
# Perceptron 의 정답률 : 0.6388888888888888
# QuadraticDiscriminantAnalysis 의 정답률 : 0.9722222222222222
# RadiusNeighborsClassifier 없는 것
# RandomForestClassifier 의 정답률 : 1.0
# RidgeClassifier 의 정답률 : 1.0
# RidgeClassifierCV 의 정답률 : 1.0
# SGDClassifier 의 정답률 : 0.6388888888888888
# SVC 의 정답률 : 0.6944444444444444
# StackingClassifier 없는 것
# VotingClassifier 없는 것