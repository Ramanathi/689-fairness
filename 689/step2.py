import numpy as np
import pulp

class PrivateFairPostprocessing:
    
    def __init__(self, Y, Z ,Yhat, p, alpha):
        '''
        Y: labels on S_2
        Z: private attributes on S_2
        Yhat: predictions of first step predictor on S_2
        p: inversion probability  (Z is A flipped with probability p)
        alpha: constraint level (\alpha_n)
        '''
        self.p = p
        self.alpha = alpha
        p_yz0, p_yz1, p_haty = self.private_quantities(Y, Z, Yhat)
        p_yz = self.compute_pyz(Y,Z)
        p_ya = self.compute_pya(p_yz)
        p_hatya = self.required_quantities(p_haty, p_ya, p_yz)
        p_ya1, p_ya0 = self.compute_pyayh(p_yz0,p_yz1)
        self.tildeY = self.solve_lp(p_ya0, p_ya1, p_hatya) 
    
    
    def compute_pyayh(self, p_yz0, p_yz1):
        '''
        compute P(HATY, A , Y) by inversion
        '''
        p = self.p
        p_ya0 = np.zeros(4) # P(haty, A , Y =0), p_ya0[2] = P(1,0,0), p_ya0[1] = P(0,1,0)
        p_ya1 = np.zeros(4) # P(haty, A , Y =1)
        a = np.array([[ p,1-p ], [ 1-p,p ]])

        b = np.array([p_yz0[2],p_yz0[3]])
        p_1a0 = np.linalg.solve(a, b) 
        
        b = np.array([p_yz1[2],p_yz1[3]])
        p_1a1 = np.linalg.solve(a, b) 
        
        b = np.array([p_yz0[0],p_yz0[1]])
        p_0a0 = np.linalg.solve(a, b)
        
        b = np.array([p_yz1[0],p_yz1[1]])
        p_0a1 = np.linalg.solve(a, b) 
        
        p_ya0[0:2] = p_0a0
        p_ya1[0:2] = p_0a1
        p_ya1[2:4] = p_1a1
        p_ya0[2:4] = p_1a0
        
        return p_ya0, p_ya1
    
    
    def compute_pyz(self, Y, Z):
        '''
        Compute P(Y=y,Z=a)
        '''
        p_yz = np.zeros(4) # P( Z , Y), p_yz[2] = P(1,0), p_yz[1] = P(0,1)

        for i in range(len(Y)):
            p_yz[Z[i] + 2*Y[i]] += 1

        for i in range(4):
            p_yz[i] /= len(Y)
            
        return p_yz
    
    
    def compute_pya(self, p_yz):
        '''
        Compute P(Y=y,A=a) by inversion
        '''
        p = self.p
        a = np.array([[ p,1-p ], [ 1-p,p ]])
        
        b = np.array([p_yz[0],p_yz[1]])
        p_0a = np.linalg.solve(a, b) #p_0a = P(Y=0,A=a)
        
        b = np.array([p_yz[2],p_yz[3]])
        p_1a = np.linalg.solve(a, b) #p_1a = P(Y=1,A=a)

        p_ya = np.concatenate((p_0a, p_1a))
        return p_ya 
    
    
    def private_quantities(self, Y, Z, Yhat):
        '''
        Assumed Z and Y are binary
        '''
        p_yz0 = np.zeros(4) # P(haty, Z , Y =0), p_ya0[2] = P(1,0,0), p_ya0[1] = P(0,1,0)
        p_yz1 = np.zeros(4) # P(haty, Z , Y =1)
        p_haty = np.zeros(4) # P(haty=1| Z , Y)
        counts_haty = np.zeros(4)

        for i in range(len(Y)):
            if Y[i] == 1:
                p_yz1[Z[i] + 2*Yhat[i]] += 1
            else:
                p_yz0[Z[i] + 2*Yhat[i]] += 1

            p_haty[Z[i] + 2*Y[i]] += Yhat[i]
            counts_haty[Z[i] + 2*Y[i]] += 1

        for i in range(4):
            p_yz0[i] /= len(Y)
            p_yz1[i] /= len(Y)
            p_haty[i] = p_haty[i] / counts_haty[i]
        return p_yz0, p_yz1, p_haty
    
    
    def required_quantities(self, p_haty, p_ya, p_yz):
        '''
        Compute P(Yhat | Y=y, A =a)
        '''
        p = self.p
        a0 = np.array([ p *p_ya[2]/p_yz[2], (1-p)*p_ya[3]/p_yz[2] ])
        a1 = np.array([ (1-p) *p_ya[2]/p_yz[3], p*p_ya[3]/p_yz[3] ])

        a = np.array([a0,a1])
        b = np.array([p_haty[2],p_haty[3]])
        p_haty1 = np.linalg.solve(a, b) #p_haty1[0] = P(Yhat=1|Y=1, A=0)
        
        a0 = np.array([ p *p_ya[0]/p_yz[0], (1-p)*p_ya[1]/p_yz[0] ])
        a1 = np.array([ (1-p) *p_ya[0]/p_yz[1], p*p_ya[1]/p_yz[1] ])

        a = np.array([a0,a1])
        b = np.array([p_haty[0],p_haty[1]])
        p_haty0 = np.linalg.solve(a, b) #p_haty0[1] = P(Yhat=1|Y=0, A=1)
        p_haty = np.zeros(4)
        p_haty[0:2] = p_haty0
        p_haty[2:4] = p_haty1
        return p_haty
        
        
        
    def solve_lp(self, p_ya0, p_ya1, p_haty):
        '''
        Solve the LP and return the solution
        '''
        problem = pulp.LpProblem("fair-postprocess",pulp.LpMinimize)
        p_tildey = pulp.LpVariable.dicts("tildeY",list(range(4)),0,1,cat="Continuous")
        # c1 for y=0 and c2 for y=1, EO constraints
        p = self.p
        c11 = (p*p_tildey[1] + (1-p)*p_tildey[0])*(1-p_haty[1]) + (p*p_tildey[3] + (1-p)*p_tildey[2])*(p_haty[1]) - ((p*p_tildey[0] + (1-p)*p_tildey[1])*(1-p_haty[0]) + (p*p_tildey[2] + (1-p)*p_tildey[3])*(p_haty[0])) <=  self.alpha
        c12 = -((p*p_tildey[1] + (1-p)*p_tildey[0])*(1-p_haty[1]) + (p*p_tildey[3] + (1-p)*p_tildey[2])*(p_haty[1])) + (p*p_tildey[0] + (1-p)*p_tildey[1])*(1-p_haty[0]) + (p*p_tildey[2] + (1-p)*p_tildey[3])*(p_haty[0]) <=  self.alpha
        c21 = (p*p_tildey[1] + (1-p)*p_tildey[0])*(1-p_haty[3]) + (p*p_tildey[3] + (1-p)*p_tildey[2])*(p_haty[3]) - ((p*p_tildey[0] + (1-p)*p_tildey[1])*(1-p_haty[2]) + (p*p_tildey[2] + (1-p)*p_tildey[3])*(p_haty[2])) <=  self.alpha
        c22 = -((p*p_tildey[1] + (1-p)*p_tildey[0])*(1-p_haty[3]) + (p*p_tildey[3] + (1-p)*p_tildey[2])*(p_haty[3])) + (p*p_tildey[0] + (1-p)*p_tildey[1])*(1-p_haty[2]) + (p*p_tildey[2] + (1-p)*p_tildey[3])*(p_haty[2]) <=  self.alpha
        problem += c11
        problem += c12
        problem += c21
        problem += c22

        problem += (p_ya1[0]-p_ya0[0])*(p*p_tildey[0] + (1-p) * p_tildey[1]) + (p_ya1[1]-p_ya0[1])*(p*p_tildey[1] + (1-p) * p_tildey[0]) + (p_ya1[2]-p_ya0[2])*(p*p_tildey[2] + (1-p) * p_tildey[3]) + (p_ya1[3]-p_ya0[3])*(p*p_tildey[3] + (1-p) * p_tildey[2])
        problem.solve()
        solution = [p_tildey[i].varValue for i in range(4)]
        return solution
    
    def predict_batch(self, Yhat, A):

        Ytilde = []
        for i in range(len(Yhat)):
            prediction = np.random.binomial(1, self.p * self.tildeY[A[i] + 2*Yhat[i]] + (1-self.p) *self.tildeY[1-A[i] + 2*Yhat[i]] , 1)[0]
            Ytilde = Ytilde + [prediction]
        return Ytilde