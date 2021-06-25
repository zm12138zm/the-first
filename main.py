import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def method():
    x = pd.read_csv('train.csv')
    df = pd.DataFrame(x)
    z = pd.read_csv('test.csv')
    DF = pd.DataFrame(z)
    x_train = df.drop(['ID', 'CLASS'], axis=1)
    x_test = DF.drop(columns=['ID'])
    y_train = df['CLASS']
    knn = KNeighborsClassifier(n_neighbors=3)
    param_dict={'n_neighbors':[1,3,5,6,7,8,10]}
    knn = GridSearchCV(knn,param_grid=param_dict,cv=5)
    knn.fit(x_train, y_train)
    predict = knn.predict(x_test)
    data = pd.DataFrame(predict, columns=['CLASS'])
    data.to_csv("submission.csv")
