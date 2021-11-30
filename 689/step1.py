from sklearn.model_selection import train_test_split
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from sklearn.linear_model import LogisticRegression
from utils import *
import math


class PrivateFairPreprocessing:
    def __init__(self, Y, A, X_scaled, exp, val_size):
        X_train, X_test, Y_train, self.Y_test, A_train, A_test = train_test_split(X_scaled, Y, A, test_size = 0.25, stratify=Y)

        X_train, X_val, self.Y_train, self.Y_val, A_train, A_val = train_test_split(X_train, Y_train, A_train, test_size = val_size, stratify=Y_train)
        

        self.exp = exp
        self.X_train = X_train.reset_index(drop=True)
        self.A_train = A_train.reset_index(drop=True)
        self.X_test = X_test.reset_index(drop=True)
        self.A_test = A_test.reset_index(drop=True)
        self.X_val = X_val.reset_index(drop=True)
        self.A_val = A_val.reset_index(drop=True)
    
    def compute(self):
        # compute Z from A
        epsilon = 10**self.exp
        pi = math.exp(epsilon) / (math.exp(epsilon)  +1)
        self.A_train = LDP(pi, self.A_train)
        self.A_val   = LDP(pi, self.A_val) 

        step1 = ExponentiatedGradient(LogisticRegression(solver='liblinear', fit_intercept=True),
                                        constraints=EqualizedOdds(),
                                        eps= 0.001,
                                        max_iter=50,
                                        eta0 = 0.5)
        step1.fit(self.X_train, self.Y_train, sensitive_features=self.A_train)
        
        return step1.predict(self.X_test), self.A_test, self.Y_test, step1.predict(self.X_val), self.A_val, self.Y_val, pi