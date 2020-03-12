import pickle as pkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
import random

import KMM
import model



num_exp = 1
epochs = 1000


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
yA_lab = pd.DataFrame()
yB_lab = pd.DataFrame()


corruption_level = .1
num_act = [3]
per = 1
idx_act = 0

act_disc = []


for numAct in num_act :
    for ne in range(num_exp):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
    
        
            #### Choose activities ####

            act_A = np.array(random.sample(range(7), 7)).astype(float)
            act_B = np.array(random.sample(range(7), numAct)).astype(float)
            rest = np.setdiff1d(act_A, act_B)

            x_act_A = pd.DataFrame()
            x_act_B = pd.DataFrame()
            x_lab_A = pd.DataFrame()
            x_lab_B = pd.DataFrame()
            
            ## Discovery Activities ##
            
            for i in rest:
                Br = yB == i
                re = XB[Br]
                labr = pd.DataFrame(yB[Br]).reset_index(drop=True)
                tempr = "dfr"+str(i) 
                vars()[tempr] = pd.DataFrame(re).reset_index(drop=True)
                r,c = vars()[tempr].shape

                rest_act = pd.concat([rest_act, vars()[tempr]], ignore_index = True)
                rest_lab = pd.concat([rest_lab, labr], ignore_index = True)
        

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


            for e in range(epochs):
                #Configure Inputs

                HA_train, HA_test, yA_train, yA_test = train_test_split(HAtrain, yA_lab, test_size=0.2, random_state= 0)
                HB_train, HB_test, yB_train, yB_test = train_test_split(HBtrain, yB_lab, test_size=0.2, random_state= 0)

                fake_A = HA_train*np.random.binomial(size=HA_train.shape, n=1, p=1-corruption_level)
                fake_B = HB_train*np.random.binomial(size=HB_train.shape, n=1, p=1-corruption_level)

                train_A = HA_train
                train_B = HB_train

                # ----------------------
                #  Run optimizers
                # ----------------------

                _ = sess.run(d_train_opt, feed_dict={DA_real: train_A, DB_real: train_B, input_zA: fake_A, input_zB: fake_B})
                _ = sess.run(g_train_opt, feed_dict={input_zA:fake_A , input_zB:fake_B})

                #----------------
                # Domain A and B
                #----------------

                train_loss_d = sess.run(d_loss, {DA_real: train_A, DB_real: train_B, input_zA: fake_A, input_zB: fake_B})
                train_loss_g = g_loss.eval({input_zA: fake_A , input_zB:fake_B})


                saver.save(sess, './checkpoints/generator.ckpt')


                    # Save losses to view after training
                    #losses.append((train_loss_d, train_loss_g))

            #------------------
            # Transfer Process
            #------------------

            saver = tf.train.Saver(var_list=G_vars)
            with tf.Session() as sess:
                saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

                gen_B = rest_act
                gen_A = np.random.uniform(0, 1, size=(gen_B.shape))

                HA_new = pd.DataFrame(np.concatenate((np.array(HA_test), gen_B), axis=0))
                HB_new = pd.DataFrame(np.concatenate((np.array(HB_test), gen_A), axis=0))

                fake_A2 = HA_new*np.random.binomial(size=HA_new.shape, n=1, p=1-.1)
                fake_B2 = HB_new*np.random.binomial(size=HB_new.shape, n=1, p=1-.1)


                fake_A2B = sess.run(generatorA(input_zA, DA_size, n_units=gA_hidden_size, alpha=alpha, reuse=True),
                               feed_dict={input_zA: fake_A2})

                fake_B2A = sess.run(generatorB(input_zB, DB_size, n_units=gB_hidden_size, alpha=alpha, reuse=True),
                               feed_dict={input_zB: fake_B2})


                recov_A = sess.run(generatorA(input_zA, DA_size, n_units=gA_hidden_size, alpha=alpha, reuse=True),
                               feed_dict={input_zA: fake_B2A})

                recov_B = sess.run(generatorB(input_zB, DB_size, n_units=gB_hidden_size, alpha=alpha, reuse=True),
                               feed_dict={input_zB: fake_A2B})
                


