from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_iris
import warnings

warnings.filterwarnings('ignore')

dataset = load_iris()
x = dataset.data
y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=66)

kfold = KFold(n_splits=5, shuffle=True)
allAlgorithm = all_estimators(type_filter = 'classifier')

for (name, algorithm) in allAlgorithm:
    try:
        model = algorithm()
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)
        print(name,'의 정답률 : \n',scores)
    except:
        print(name, '없는 것')

# import sklearn
# print(sklearn.__version__) # 0.23.2
# AdaBoostClassifier 의 정답률 : 
#  [0.91666667 0.875      0.95833333 0.91666667 1.        ]
# BaggingClassifier 의 정답률 : 
#  [0.91666667 0.95833333 0.95833333 0.95833333 0.875     ]
# BernoulliNB 의 정답률 :
#  [0.29166667 0.20833333 0.33333333 0.20833333 0.33333333]
# CalibratedClassifierCV 의 정답률 : 
#  [0.95833333 0.79166667 0.875      1.         0.83333333]
# CategoricalNB 의 정답률 :
#  [0.95833333 1.         0.95833333 0.83333333 0.95833333]
# ComplementNB 의 정답률 : 
#  [0.79166667 0.70833333 0.66666667 0.54166667 0.625     ]
# DecisionTreeClassifier 의 정답률 :
#  [0.95833333 0.95833333 0.91666667 0.95833333 0.875     ]
# DummyClassifier 의 정답률 :
#  [0.375      0.375      0.45833333 0.33333333 0.20833333]
# ExtraTreeClassifier 의 정답률 : 
#  [0.875      1.         0.91666667 0.91666667 0.91666667]
# ExtraTreesClassifier 의 정답률 : 
#  [0.91666667 0.95833333 0.95833333 0.95833333 0.95833333]
# GaussianNB 의 정답률 :
#  [1.         0.91666667 0.91666667 0.95833333 0.95833333]
# GaussianProcessClassifier 의 정답률 : 
#  [0.95833333 1.         0.95833333 0.95833333 0.91666667]
# GradientBoostingClassifier 의 정답률 : 
#  [1.         0.95833333 0.875      0.91666667 0.95833333]
# HistGradientBoostingClassifier 의 정답률 : 
#  [0.91666667 0.91666667 0.95833333 1.         0.95833333]
# KNeighborsClassifier 의 정답률 :
#  [0.95833333 0.91666667 0.95833333 0.95833333 1.        ]
# LabelPropagation 의 정답률 :
#  [0.95833333 0.95833333 0.95833333 0.95833333 0.95833333]
# LabelSpreading 의 정답률 :
#  [0.91666667 0.875      1.         1.         0.95833333]
# LinearDiscriminantAnalysis 의 정답률 : 
#  [0.95833333 1.         1.         0.95833333 1.        ]
# LinearSVC 의 정답률 : 
#  [0.95833333 0.95833333 0.95833333 0.95833333 0.91666667]
# LogisticRegression 의 정답률 : 
#  [0.91666667 1.         0.91666667 0.95833333 1.        ]
# LogisticRegressionCV 의 정답률 : 
#  [0.91666667 0.95833333 0.91666667 1.         0.95833333]
# MLPClassifier 의 정답률 : 
#  [0.95833333 1.         1.         1.         0.95833333]
# MultinomialNB 의 정답률 :
#  [0.79166667 0.91666667 0.91666667 0.91666667 0.95833333]
# NearestCentroid 의 정답률 :
#  [0.91666667 0.91666667 0.91666667 0.875      0.91666667]
# NuSVC 의 정답률 : 
#  [0.95833333 0.91666667 1.         1.         0.95833333]
# PassiveAggressiveClassifier 의 정답률 :
#  [0.75       0.75       0.875      1.         0.95833333]
# Perceptron 의 정답률 :
#  [0.58333333 0.58333333 1.         0.79166667 0.875     ]
# QuadraticDiscriminantAnalysis 의 정답률 : 
#  [1.         1.         0.95833333 1.         0.91666667]
# RadiusNeighborsClassifier 의 정답률 :
#  [1.         0.95833333 0.95833333 0.91666667 0.95833333]
# RandomForestClassifier 의 정답률 : 
#  [1.         0.91666667 1.         0.91666667 0.875     ]
# RidgeClassifier 의 정답률 :
#  [0.875      0.75       0.875      0.83333333 0.79166667]
# RidgeClassifierCV 의 정답률 :
#  [0.79166667 0.875      0.75       0.95833333 0.91666667]
# SGDClassifier 의 정답률 : 
#  [0.66666667 0.58333333 0.91666667 0.79166667 0.875     ]
# SVC 의 정답률 :
#  [0.95833333 0.95833333 1.         1.         0.95833333]