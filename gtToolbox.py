import sys
import csv
import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt


##
#
#   The user inputs a .xml file that contains the values for each
#   resource and the action sets of each player
#

A = [3,2.99,0.01,0.01]
B = [[0,1],[0,2],[0,3]]

def gairing(j,k):
    num2 = np.sum(1/factorial(np.arange(j,k)))
    num1 = 1/(k-1)/factorial(k-1)
    return factorial(j-1)*(num1+num2)/(num1+np.sum(1/factorial(np.arange(1,k))))

def marginal_contribution(j,k):
    return 1*(j==1)

def equal_share(j,k):
    return 1/j

class GameObject:
    def __init__(self, res, act):
        self.num_players = len(act)
        self.resources = res
        self.action_sets = act
        self.dist_rule = None
        self.lambda_mu = {}

    def generate_table(self, dist_rule):
        print('GameObject:generate_table: using function \'%s\' as distribution rule.' % dist_rule.__name__)
        self.dist_rule = dist_rule
        
        table_shape = [1]
        for i in range(self.num_players):
            table_shape[0] = table_shape[0] * len(self.action_sets[i])
        table_shape.append(self.num_players+1)

        self.table = np.zeros(table_shape)
        for k in range(np.shape(self.table)[0]):
            divide_param = 1
            res_idx = np.zeros(self.num_players, dtype=np.int)
            bin_vals = np.zeros(self.num_players)
            for i in range(self.num_players):
                act_length = len(self.action_sets[i])
                act_idx = int(k / divide_param) % act_length
                res_idx[i] = self.action_sets[i][act_idx]                
                bin_vals[i] = self.resources[res_idx[i]]
                divide_param = divide_param * act_length

            for i in range(self.num_players):
                overlap = np.count_nonzero(res_idx == res_idx[i])
                self.table[k,i] = bin_vals[i] * self.dist_rule(overlap, self.num_players)
            self.table[k,self.num_players] = np.sum(self.resources[np.unique(res_idx)])

    def construct_lambda_mu(self):
        for k in range(np.shape(self.table)[0]):
            s_divide_param = 1
            s = np.zeros(self.num_players, dtype=np.int)
            for i in range(self.num_players):
                act_length = len(self.action_sets[i])
                s[i] = int(k / s_divide_param) % act_length
                s_divide_param = s_divide_param * act_length
            for l in range(np.shape(self.table)[0]):
                s_star_divide_param = 1
                sum_utils = 0
                s_star = np.zeros(self.num_players, dtype=np.int)
                for j in range(self.num_players):
                    act_length_star = len(self.action_sets[j])
                    s_star[j] = int(l / s_star_divide_param) % act_length_star
                    s_star_divide_param = s_star_divide_param * act_length_star
                for i, s_star_i in enumerate(s_star):
                    tmp = np.copy(s)
                    tmp[i] = s_star_i
                    m = 0
                    reconstruct_param = 1
                    for p in range(self.num_players):
                        m = m + reconstruct_param * tmp[p]
                        reconstruct_param = reconstruct_param*len(self.action_sets[p])
                    sum_utils = sum_utils + self.table[m,i]
                slope = "%.8f" % (self.table[l,self.num_players] / self.table[k,self.num_players])
                y_int = sum_utils / self.table[k,self.num_players]
                # print("%.8f >= %dL - %dU" % (sum_utils, self.table[l,self.num_players], self.table[k,self.num_players]))
                if slope in self.lambda_mu:
                    self.lambda_mu[slope] = np.min([y_int, self.lambda_mu[slope]])
                else:
                    self.lambda_mu[slope] = y_int
                    
    def robust_poa(self):
        xMax = 100
        x = np.arange(0,xMax,0.001)
        slopesAndYInts = []
        maxSlope = -1
        for slope in sorted(self.lambda_mu, key=self.lambda_mu.get):
            fSlope = float(slope)
            if fSlope > maxSlope:
                slopesAndYInts.append((fSlope, self.lambda_mu[slope]))
                maxSlope = fSlope

        tableOfVals = np.zeros((len(slopesAndYInts), np.size(x)))
        for idx, (slope,y_int) in enumerate(slopesAndYInts):
            tableOfVals[idx,:] = slope*x-y_int
        maxVals = np.max(tableOfVals, axis=0)
        robustPoAs = x/(1+maxVals)
        bestPoA = np.max(robustPoAs)
        bestPoAIdx = np.argmax(robustPoAs)
        lam = bestPoAIdx/np.size(x)*xMax
        mu = maxVals[bestPoAIdx]

        return (lam, mu, bestPoA)
        
    def print_table(self):
        print('GameObject:print_table():\n', self.table)

def main(args):
    res = []
    act = []
    
    try:
        with open(args[1]) as csvfile:
            for line in csv.reader(csvfile):
                act.append(line)
        with open(args[2]) as csvfile:
            for line in csv.reader(csvfile):
                res = line
    except IndexError:
        print('Please provide two .csv files to be parsed.')
        return

    act = [[int(i) for i in act_set] for act_set in act]
    res = np.array(res, dtype=np.float)

    print('__main__: array \'res\':', res)
    print('__main__: list \'act\':', act)
    
    g = GameObject(res, act)
    g.generate_table(equal_share)
    g.construct_lambda_mu()
    print(g.robust_poa())
    
    g.generate_table(gairing)
    g.construct_lambda_mu()
    print(g.robust_poa())
    
    g.generate_table(marginal_contribution)
    g.construct_lambda_mu()
    print(g.robust_poa())
    
    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))
