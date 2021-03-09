from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_breast_cancer
import warnings

warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
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