from sklearn.preprocessing import LabelEncoder,StandardScaler
import pandas as pd
import shap, copy
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from step2 import *
from step1 import *

exponents = [-2,-1,-0.75,-0.5,-0.25,-0.1,0.1,0.25,0.5,1,1.5,2]
X_raw, Y = shap.datasets.adult()
A = X_raw["Sex"]
X = X_raw.drop(labels=['Sex'],axis = 1)
X = pd.get_dummies(X)
X_scaled = StandardScaler().fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
Y = LabelEncoder().fit_transform(Y)

def Algo(only_step1, val_size):
    error_results, fair_results = [], []
    for exp in exponents:
        errors = []
        discs = []
        print("for epsilong = ", 10**exp)
        for _ in range(0,5):

            predictions_test, A_test, Y_test, predictions_val, A_val, Y_val, pi = PrivateFairPreprocessing(Y, A, X_scaled, exp, val_size).compute()

            if not only_step1:
                fair = PrivateFairPostprocessing(Y_val, A_val, predictions_val, pi, 0.0001)
                predictions = fair.predict_batch(predictions_test, copy.deepcopy(A_test))
            else:
                predictions = predictions_test
        
            error =accuracy_score(predictions,Y_test)
            disc = max(EO(Y_test,A_test, predictions ,0), EO(Y_test,A_test, predictions ,1))
            errors = errors + [error]
            discs  = discs + [disc] 

        error_results = error_results + [errors]
        fair_results  = fair_results + [discs]
    
    return error_results, fair_results

# only step 1
error_results_1, fair_results_1       = Algo(only_step1 = True, val_size = 0.1)

# both step 1 and step 2 
error_results_12, fair_results_12     = Algo(only_step1 = False, val_size = 0.5)

avgs_error_12, error_bars_error_12  = getAvgs_and_error_bars(error_results_12)
    
avgs_fair_12, error_bars_fair_12    = getAvgs_and_error_bars(fair_results_12)
    
avgs_error_1, error_bars_error_1    = getAvgs_and_error_bars(error_results_1)
    
avgs_fair_1, error_bars_fair_1      = getAvgs_and_error_bars(fair_results_1)

fig, axs = plt.subplots(2)
axs[0].errorbar(exponents,avgs_fair_1 , yerr=error_bars_fair_1, label='step 1')
axs[0].errorbar(exponents,avgs_fair_12 , yerr=error_bars_fair_12, label='2-step')

axs[0].legend(loc='upper right')
axs[0].set_ylabel(r'discrimination')
axs[0].set_xlabel(f'log($\\epsilon$)')

axs[1].errorbar(exponents,1- np.asarray(avgs_error_1) , yerr=error_bars_error_1, label='step 1')
axs[1].errorbar(exponents,1- np.asarray(avgs_error_12) , yerr=error_bars_error_12, label='2-step')

axs[1].legend(loc='lower right')
axs[1].set_ylabel(r'error')
axs[1].set_xlabel(f'log($\\epsilon$)')
plt.show()