import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score
from sklearn.svm import SVC
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.spatial.distance import cdist
import sklearn as sk
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifier, RidgeClassifierCV
from sklearn.model_selection import cross_val_predict
from os.path import basename
from cvxopt import matrix, solvers
from TLDA import is_pos_def, one_hot
import random



## == Parameters == ##

num_exp = 1
num_body = 3
fs_exp = []
num_feat = [1]
per = [.2]
num_act = [19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2]
p = .1

labels_HA_train = []
labels_HB_train = []
labels_HA_test = []
labels_HB_test = []
HAtrain =  pd.DataFrame()
HBtrain =  pd.DataFrame()
HA_test =  pd.DataFrame()
HB_test =  pd.DataFrame()
new_act_HB = pd.DataFrame()
new_act_HA = pd.DataFrame()
unknown_act_tar =  pd.DataFrame()
unknown_lab_tar =  pd.DataFrame()
unknown_act_src =  pd.DataFrame()
unknown_lab_src =  pd.DataFrame()
rest_lab =  pd.DataFrame()
rest_act =  pd.DataFrame()
rest_train_act = pd.DataFrame()
rest_train_lab = pd.DataFrame()
yA_lab = pd.DataFrame()
yB_lab = pd.DataFrame()

houseB = r'dsads.csv'
data_B = pd.read_csv(houseB ,sep=',',header=0)

# 19 activities #

RA = data_B.iloc[:,82:163]
LA = data_B.iloc[:,163:244]

RL = data_B.iloc[:,244:325]
LL = data_B.iloc[:,325:406]

T = data_B.iloc[:,1:82]

yA = data_B.iloc[:,0]
yB = data_B.iloc[:,0]


Body1 = [RA, RL, RA]
Body2 = [LA, LL, T]
Body_exp = ['RA - LA', 'RL - LL', 'RA - T']
y_bodyA = [yA, yA, yA]
y_bodyB = [yB, yB, yB]


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1, pt=p):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.pt = pt

    def fit(self, Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = np.dot(A.T, K)
        Z /= np.linalg.norm(Z, axis=0)
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt,pt):
        '''
        Transform Xs and Xt, then make predictions on target using 1NN
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted_labels on the target domain
        '''
        Xs_new, Xt_new = self.fit(Xs, Xt)


            
        HA_train, HA_test, yA_train, yA_test = train_test_split(Xs_new, Ys, test_size=pt, random_state= 0)
        HB_train, HB_test, yB_train, yB_test = train_test_split(Xt_new, Yt, test_size=pt, random_state= 0)


        parameters  = {'C':[.1,1],'gamma':[.1,1]}
        smv_dism_data1 = svm.SVC(kernel='rbf', probability = True, class_weight = 'balanced')
        clf = GridSearchCV(smv_dism_data1, parameters)
        predicted_data1 = clf.fit(HA_train, yA_train)
        pred_label2 = predicted_data1.predict(Xt_new)
        rs2 = recall_score(yB_test, pred_label2, average='micro') 
        ps2 = precision_score(yB_test, pred_label2, average='micro') 
        Fab = ((2*(rs2*ps2)/(rs2+ps2)))

        parameters  = {'C':[.1,1],'gamma':[.1,1]}
        smv_dism_data1 = svm.SVC(kernel='rbf', probability = True, class_weight = 'balanced')
        clf = GridSearchCV(smv_dism_data1, parameters)
        predicted_data1 = clf.fit(HA_train, yA_train)
        pred_label2 = predicted_data1.predict(Xt_new)
        rs2 = recall_score(yB_test, pred_label2, average='macro') 
        ps2 = precision_score(yB_test, pred_label2, average='macro') 
        Fba = ((2*(rs2*ps2)/(rs2+ps2)))
            
        return Fab, Fba


if __name__ == '__main__':
    
    for numAct in num_act :
        for ne in range(num_exp):
            for numf in num_feat:
                for num_b in range(num_body):
                    for p in per:
                        Body = Body_exp[num_b]

                        XA = Body1[num_b]
                        XB = Body2[num_b]

                        yA = y_bodyA[num_b]
                        yB = y_bodyB[num_b]

                        ma, na = XA.shape
                        mb, nb = XB.shape


                        if ma < mb:
                            id_df = pd.DataFrame(XB).sample(n=ma, replace=False).index
                            X_B = pd.DataFrame(XB).iloc[id_df,:]
                            y_label_B = pd.DataFrame(yB).iloc[id_df,:]
                            X_A = XA
                            y_label_A = yA
                        else:
                            id_df = pd.DataFrame(XA).sample(n=mb, replace=False).index
                            X_A = pd.DataFrame(XA).iloc[id_df,:]
                            y_label_A = pd.DataFrame(yA).iloc[id_df,:]
                            X_B = XB
                            y_label_B = yB

                        #### Choose activities ####

                        act_A = np.array(random.sample(range(19), 19)).astype(float)
                        act_B = np.array(random.sample(range(19), numAct)).astype(float)
                        rest = np.setdiff1d(act_A, act_B)

                        x_act_A = pd.DataFrame()
                        x_act_B = pd.DataFrame()
                        x_lab_A = pd.DataFrame()
                        x_lab_B = pd.DataFrame()

                        ## Discovery Activities - unknown ##

                        for i in rest:
                            Br = yB == i
                            re = XB[Br]
                            labr = pd.DataFrame(yB[Br]).reset_index(drop=True)
                            tempr = "dfr"+str(i) 
                            vars()[tempr] = pd.DataFrame(re).reset_index(drop=True)
                            r,c = vars()[tempr].shape

                            rest_act = pd.concat([rest_act, vars()[tempr]], ignore_index = True)
                            rest_lab = pd.concat([rest_lab, labr], ignore_index = True)


                        ## Discovery Activities - train unknown ##

                        for i in rest:
                            Ar = yA == i
                            reA = XA[Ar]
                            labrA = pd.DataFrame(yA[Ar]).reset_index(drop=True)
                            temprA = "dfrA"+str(i) 
                            vars()[temprA] = pd.DataFrame(reA).reset_index(drop=True)
                            r,c = vars()[temprA].shape

                            rest_train_act = pd.concat([rest_train_act, vars()[temprA]], ignore_index = True)
                            rest_train_lab = pd.concat([rest_train_lab, labrA], ignore_index = True)


                        #Separate activities
                        #House A
                        for i in act_A:
                            A = yA == i
                            d = XA[A]
                            lab = pd.DataFrame(yA[A]).reset_index(drop=True)
                            temp = "df"+str(i) 
                            vars()[temp] = pd.DataFrame(d).reset_index(drop=True)
                            r,c = vars()[temp].shape

                            x_act_A = pd.concat([x_act_A, vars()[temp]], ignore_index = True)
                            x_lab_A = pd.concat([x_lab_A, lab], ignore_index = True)

                        #Separate activities
                        #House B
                        for i in act_B:
                            B = yB == i
                            dd = XB[B]
                            labb = pd.DataFrame(yB[B]).reset_index(drop=True)
                            temp = "df"+str(i) 
                            vars()[temp] = pd.DataFrame(dd).reset_index(drop=True)
                            r,c = vars()[temp].shape

                            x_act_B = pd.concat([x_act_B, vars()[temp]], ignore_index = True)
                            x_lab_B = pd.concat([x_lab_B, labb], ignore_index = True)



                        if len(act_A) < len(act_B):
                            act_array = np.unique(act_B)
                        else:
                            act_array = np.unique(act_A)



                        for la in act_array:
                            #filter per activity
                            #labels from the source domain
                            A = x_lab_A == la
                            dA = x_act_A[np.array(A.iloc[:,0])]
                            lab_A = pd.DataFrame(x_lab_A[A.iloc[:,0]]).reset_index(drop=True)
                            tempA = "dfA"+str(la) 
                            vars()[tempA] = pd.DataFrame(dA).reset_index(drop=True)
                            ra,ca = vars()[tempA].shape

                            #labels from the target domain
                            B = x_lab_B == la
                            dB = x_act_B[np.array(B.iloc[:,0])]
                            lab_B = pd.DataFrame(x_lab_B[B.iloc[:,0]]).reset_index(drop=True)
                            tempB = "dfB"+str(la) 
                            vars()[tempB] = pd.DataFrame(dB).reset_index(drop=True)
                            rb,cb = vars()[tempB].shape


                            if ra == 0: #activity missing from source domain
                                if rb > 0:
                                    unknown_act_tar = pd.concat([unknown_act_tar,vars()[tempB]], ignore_index = True)
                                    unknown_lab_tar = pd.concat([unknown_lab_tar,lab_B], ignore_index = True)

                            if rb == 0: #activity missing from source target
                                if ra > 0:
                                    unknown_act_src = pd.concat([unknown_act_src,vars()[tempA]], ignore_index = True)
                                    unknown_lab_src = pd.concat([unknown_lab_src,lab_A], ignore_index = True)

                            if ra < rb:
                                if ra > 0:
                                    HBtrain = pd.concat([HBtrain,pd.DataFrame(vars()[tempB]).sample(ra)], ignore_index = True) 
                                    yB_lab = pd.concat([yB_lab, lab_B.iloc[0:ra,0]], ignore_index = True )

                                    HAtrain = pd.concat([HAtrain,vars()[tempA]], ignore_index = True) 
                                    yA_lab = pd.concat([yA_lab, lab_A], ignore_index = True )
                                #else:
                                    #HB_train = pd.concat([HB_train,vars()[tempB]], ignore_index = True) 
                                    #yB_lab = pd.concat([yB_lab, lab_B], ignore_index = True )

                            else:
                                if rb > 0:
                                    HAtrain = pd.concat([HAtrain,pd.DataFrame(vars()[tempA]).sample(rb)], ignore_index = True)
                                    yA_lab = pd.concat([yA_lab, lab_A.iloc[0:rb,0]], ignore_index = True )

                                    HBtrain = pd.concat([HBtrain,vars()[tempB]], ignore_index = True) 
                                    yB_lab = pd.concat([yB_lab, lab_B], ignore_index = True )
                                #else:
                                    #HA_train = pd.concat([HA_train,vars()[tempA]], ignore_index = True) 
                                    #yA_lab = pd.concat([yA_lab, lab_A], ignore_index = True )

                        tca = TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)
                        Fab, Fba = tca.fit_predict(HAtrain, yA_lab, HBtrain, yB_lab,pt = p)
                        
print(Fab)
print(Fba)
