import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, lr, iteration, loss, epsilon=0.0001, w=[], max=[], min=[], tr_times=0):
        self.lr = lr
        self.iteration = iteration
        self.epsilon = epsilon
        self.w = w
        self.max = max
        self.min = min
        self.loss = loss
        self.tr_times = tr_times

    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))

    def grad(self, w, x, y):
        return ((y - self.sigmoid(x @ w)).T @ x).T

    def fit(self, train_x, train_y):

        m = train_x.shape[0]
        d = train_x.shape[1]

        w = np.ones((d + 1, 1))

        for i in range(d):
            self.max.append(train_x.iloc[:, i].max())
            self.min.append(train_x.iloc[:, i].min())
            train_x.iloc[:, i] = (
                train_x.iloc[:, i] - self.min[i]) / (self.max[i] - self.min[i])

        train_x = np.array(train_x)
        train_x = np.c_[train_x, np.ones(shape=(m, 1))]

        train_y = np.array(train_y).reshape(len(train_y), 1)

        l1 = 0
        for i in range(m):
            l1 += np.log2(1 + np.exp((np.dot(train_x[i], w)[0]))
                          ) - train_y[i] * (np.dot(train_x[i], w)[0])

        counter = 0

        while True:
            counter += 1
            dl = self.grad(w, train_x, train_y)
            w = w + self.lr * dl

            self.lr = 0.95 * self.lr

            l2 = 0
            for i in range(m):
                l2 += np.log2(1 + np.exp((np.dot(train_x[i], w)[0]))) - train_y[i] * (
                    np.dot(train_x[i], w)[0])

            self.loss.append(l2/m)
            print(counter, l2, len(self.loss))

            if abs(l2-l1) < self.epsilon and counter >= self.iteration:
                break

            l1 = l2

        self.w = w
        self.tr_times = counter

        print('train', counter, ' times')

    def predict(self, test_x):
        for i in range(test_x.shape[1]):
            test_x.iloc[:, i] = (test_x.iloc[:, i] -
                                 self.min[i]) / (self.max[i] - self.min[i])
        test_x = np.array(test_x)
        test_x = np.c_[test_x, np.ones(test_x.shape[0])]

        pre = self.sigmoid(test_x @ self.w).flatten().tolist()
        for i in range(len(pre)):
            if pre[i] > 0.5:
                pre[i] = 1
            else:
                pre[i] = 0
        return pre

    def evaluate(self, pre, test_y):
        test_y = np.array(test_y)

        assert len(pre) == len(test_y)

        counter = 0

        for i in range(len(pre)):
            if pre[i] == test_y[i]:
                counter += 1

        print('correct rate:',  counter / len(pre))


if __name__ == '__main__':
    df = pd.read_csv('loan.csv')

    df.Gender = df.Gender.map({'Male': 1, 'Female': 0})
    df.Married = df.Married.map({'Yes': 1, 'No': 0})
    df.Dependents = df.Dependents.map({'0': 0, '1': 1, '2': 2, '3+': 3})
    df.Education = df.Education.map({'Graduate': 1, 'Not Graduate': 0})
    df.Self_Employed = df.Self_Employed.map({'Yes': 1, 'No': 0})
    df.Property_Area = df.Property_Area.map(
        {'Urban': 1, 'Semiurban': 0.5, 'Rural': 0})
    df = df.fillna({'Gender': 0.5, 'Married': 0.5,
                    'Dependents': df['Dependents'].mean(),
                    'Self_Employed': df['Self_Employed'].mean(),
                    'LoanAmount': df['LoanAmount'].mean(),
                    'Loan_Amount_Term': df['Loan_Amount_Term'].mean(),
                    'Credit_History': df['Credit_History'].mean()})

    df.Loan_Status = df.Loan_Status.map({'Y': 1, 'N': 0})

    train = df.sample(frac=0.9, random_state=3, axis=0)
    test = df[~df.index.isin(train.index)]

    X_train = train.loc[:, 'Gender':'Loan_Status']
    Y_train = train.loc[:, 'Loan_Status':'Loan_Status']
    X_test = test.loc[:, 'Gender':'Loan_Status']
    Y_test = test.loc[:, 'Loan_Status':'Loan_Status']

    LR = LogisticRegression(0.05, 1000)
    LR.fit(X_train, Y_train)
