from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_diabetes
import warnings

warnings.filterwarnings('ignore')

dataset = load_diabetes()
x = dataset.data
y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=66)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x = scaler.transform(x)

allAlgorithm = all_estimators(type_filter = 'regressor')

for (name, algorithm) in allAlgorithm:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name,'의 정답률 :',r2_score(y_test,y_pred))
    except:
        print(name, '없는 것')

import sklearn
# print(sklearn.__version__) # 0.23.2

# ARDRegression 의 정답률 : 0.4987482890562275
# AdaBoostRegressor 의 정답률 : 0.37394856256320685
# BaggingRegressor 의 정답률 : 0.3465447784870761
# BayesianRidge 의 정답률 : 0.501436686384745
# CCA 의 정답률 : 0.48696409064967605
# DecisionTreeRegressor 의 정답률 : -0.20433626461140686
# DummyRegressor 의 정답률 : -0.00015425885559339214
# ElasticNet 의 정답률 : 0.11987522766332959
# ElasticNetCV 의 정답률 : 0.48941369735908513
# ExtraTreeRegressor 의 정답률 : -0.06538870412376707
# ExtraTreesRegressor 의 정답률 : 0.38275055162455507
# GammaRegressor 의 정답률 : 0.07219655012236648
# GaussianProcessRegressor 의 정답률 : -7.547010959396891
# GeneralizedLinearRegressor 의 정답률 : 0.07335459385974397
# GradientBoostingRegressor 의 정답률 : 0.38622407745463627
# HistGradientBoostingRegressor 의 정답률 : 0.28899497703380905
# HuberRegressor 의 정답률 : 0.5068530959913715
# IsotonicRegression 없는 것
# KNeighborsRegressor 의 정답률 : 0.3741821819765594
# KernelRidge 의 정답률 : 0.48022687224693383
# Lars 의 정답률 : 0.4919866521464161
# LarsCV 의 정답률 : 0.5010892359535756
# Lasso 의 정답률 : 0.46430753276688697
# LassoCV 의 정답률 : 0.4992382182931272
# LassoLars 의 정답률 : 0.36543887418957943
# LassoLarsCV 의 정답률 : 0.49519427906782476
# LassoLarsIC 의 정답률 : 0.49940515175310696
# LinearRegression 의 정답률 : 0.5063891053505036
# LinearSVR 의 정답률 : 0.14945390399691327
# MLPRegressor 의 정답률 : -0.6955272533071595
# MultiOutputRegressor 없는 것
# MultiTaskElasticNet 없는 것
# MultiTaskElasticNetCV 없는 것
# MultiTaskLasso 없는 것
# MultiTaskLassoCV 없는 것
# NuSVR 의 정답률 : 0.12527149380257419
# OrthogonalMatchingPursuit 의 정답률 : 0.3293449115305739
# OrthogonalMatchingPursuitCV 의 정답률 : 0.44354253337919736
# PLSCanonical 의 정답률 : -0.9750792277922911
# PLSRegression 의 정답률 : 0.4766139460349791
# PassiveAggressiveRegressor 의 정답률 : 0.48701182101778207
# PoissonRegressor 의 정답률 : 0.4823231874898398
# RANSACRegressor 의 정답률 : 0.21956545260160487
# RadiusNeighborsRegressor 의 정답률 : 0.14407236562185122
# RandomForestRegressor 의 정답률 : 0.3842324066339594
# RegressorChain 없는 것
# Ridge 의 정답률 : 0.49950383964954104
# RidgeCV 의 정답률 : 0.4995038396495408
# SGDRegressor 의 정답률 : 0.4961185297528524
# SVR 의 정답률 : 0.12343791188320263
# StackingRegressor 없는 것
# TheilSenRegressor 의 정답률 : 0.503544164203291
# TransformedTargetRegressor 의 정답률 : 0.5063891053505036
# TweedieRegressor 의 정답률 : 0.07335459385974397
# VotingRegressor 없는 것
# _SigmoidCalibration 없는 것