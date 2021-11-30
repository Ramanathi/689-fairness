import numpy as np
import scipy.stats as st

def EO(Y, A, Yhat, y):
    num_array = np.zeros(2)
    denom_array = np.zeros(2)
    for i in range(len(Y)):
        if Y[i] == y:
            denom_array[A[i]] += 1
            if Yhat[i] == 1:
                num_array[A[i]] += 1

    measure = num_array/denom_array
    diff = []
    for i in range(0, 2):
        for j in range(0, 2):
            diff = diff + [abs(measure[i]-measure[j])]
    return max(diff)

def LDP(pi, A):
    ''' pi being inversion probability '''
    for i in range(0,len(A)):
        coin = np.random.binomial(1, pi, 1)[0]
        if coin == 0:
            A[i] = abs(A[i]-1)
    return A

def getAvgs_and_error_bars(result, epss = 12, max_trials = 5):
    avgs, errors = np.zeros(epss), np.zeros(epss)
    for i in range(0,epss):
        arr = np.zeros(max_trials)
        for j in range(0,max_trials):
            arr[j] = result[i][j]
        avgs[i] = np.average(arr)
        interval = st.t.interval(0.95, len(arr)-1, loc=np.mean(arr), scale=st.sem(arr))
        errors[i] = (interval[1] - interval[0])/2 
    
    return avgs, errors
