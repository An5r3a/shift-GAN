import pandas as pd
import numpy as np
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
import model
import KMM

import scipy.stats as st
from scipy.spatial.distance import cdist
import sklearn as sk
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifier, RidgeClassifierCV
from sklearn.model_selection import cross_val_predict
from os.path import basename
from cvxopt import matrix, solvers
from TLDA import is_pos_def, one_hot

import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier


import ray
from ray import tune
from ray.tune import track
from ray.tune.schedulers import AsyncHyperBandScheduler




def model_inputs(real_dimA, real_dimB, zA_dim, zB_dim):
     
    #Disciminator A
    DA_real = tf.placeholder(tf.float32,(None, real_dimA), name = 'DA_real')
    #Discriminator B
    DB_real = tf.placeholder(tf.float32,(None, real_dimB), name = 'DB_real') 
    #Generator A
    inputs_zA = tf.placeholder(tf.float32,(None, zA_dim), name = 'input_zA')
    #Generator b
    inputs_zB = tf.placeholder(tf.float32,(None, zB_dim), name = 'input_zB')
    
    return DA_real, DB_real, inputs_zA, inputs_zB
    
tf.reset_default_graph()


#################################################################################################################
   #  Create our input placeholders #
#################################################################################################################

DA_real, DB_real, input_zA, input_zB = model_inputs(DA_size, DB_size, zA_size, zB_size)

''' Build the generator A network.
 g_model_A is the generator output
'''
g_model_A2B = generatorA(input_zA, DA_size, n_units=gA_hidden_size, alpha=alpha)

'''Build the generator B network.
g_model_B is the generator output
'''
g_model_B2A = generatorB(input_zB, DB_size, n_units=gB_hidden_size, alpha=alpha)

''' Generator A network 
 g_model_B is the generator input
'''
g_model_A2B2A = generatorA(g_model_B2A, DA_size, n_units=gA_hidden_size, alpha=alpha, reuse = True)

''' Generator B network 
 g_model_A is the generator input
'''
g_model_B2A2B = generatorB(g_model_A2B, DB_size, n_units=gB_hidden_size, alpha=alpha, reuse = True)


''' Define reconstrution error L2 '''

B_loss = tf.reduce_mean(tf.square(g_model_A2B2A - input_zB))
A_loss = tf.reduce_mean(tf.square(g_model_B2A2B - input_zA))


''' Define reconstrution error L1 '''

#B_loss = tf.reduce_mean(tf.abs(g_model_A2B2A - input_zB))
#A_loss = tf.reduce_mean(tf.abs(g_model_B2A2B - input_zA))


''' Build the discriminator A network.
 We'll build two of them, one for real data and one for fake data 
'''
dA_model_real, dA_logits_real = discriminatorA(DA_real, n_units=dA_hidden_size, alpha=alpha)
dA_model_fake, dA_logits_fake = discriminatorA(g_model_A2B, reuse=True, n_units=dA_hidden_size, alpha=alpha)


CA_loss = tf.reduce_mean(
                  tf.nn.sigmoid_cross_entropy_with_logits(logits=dA_logits_real, 
                                                          labels=tf.ones_like(dA_logits_real) * (1 - smooth))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dA_logits_fake, 
                                                          labels=tf.zeros_like(dA_logits_real)))

# GAN D loss
DA_loss = tf.reduce_mean(tf.log(dA_model_real + eps) + tf.log(1. - dA_model_fake + eps))

dA_loss = (DA_loss + CA_loss)



gA_loss = tf.reduce_mean(
             tf.nn.sigmoid_cross_entropy_with_logits(logits=dA_logits_fake,
                                                     labels=tf.ones_like(dA_logits_fake))) + lambda_b*B_loss


''' Build the discriminator B network.
 We'll build two of them, one for real data and one for fake data 
'''
dB_model_real, dB_logits_real = discriminatorB(DB_real, n_units=dB_hidden_size, alpha=alpha)
dB_model_fake, dB_logits_fake = discriminatorB(g_model_B2A, reuse=True, n_units=dB_hidden_size, alpha=alpha)


CB_loss = tf.reduce_mean(
                  tf.nn.sigmoid_cross_entropy_with_logits(logits=dB_logits_real, 
                                                          labels=tf.ones_like(dB_logits_real) * (1 - smooth))) + tf.reduce_mean(
                  tf.nn.sigmoid_cross_entropy_with_logits(logits=dB_logits_fake, 
                                                          labels=tf.zeros_like(dB_logits_real)))

# GAN D loss
DB_loss = tf.reduce_mean(tf.log(dB_model_real + eps) + tf.log(1. - dB_model_fake + eps))

dB_loss = DB_loss + CB_loss


gB_loss = tf.reduce_mean(
             tf.nn.sigmoid_cross_entropy_with_logits(logits=dB_logits_fake,
                                                     labels=tf.ones_like(dB_logits_fake))) + lambda_a*A_loss

d_loss = dA_loss + dB_loss
g_loss = gA_loss + gB_loss  

#################################################################################################################
   #  Optimizers #
#################################################################################################################

learning_rate = 0.01 

# Get the trainable_variables, split into G and D parts
t_vars = tf.trainable_variables()
Ad_vars = [var for var in t_vars if var.name.startswith('discriminatorA')]
Bd_vars = [var for var in t_vars if var.name.startswith('discriminatorB')]
Ag_vars = [var for var in t_vars if var.name.startswith('generatorA')]
Bg_vars = [var for var in t_vars if var.name.startswith('generatorB')]

D_vars = Ad_vars + Bd_vars
G_vars = Ag_vars + Bg_vars

d_train_opt = (tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(d_loss, var_list=D_vars))
g_train_opt = (tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(g_loss, var_list=G_vars))


#################################################################################################################
   #  Step 1: Feature Space Transformation #
#################################################################################################################

saver = tf.train.Saver(var_list=G_vars)
epochs = 2000

for per in corruption_level:
    for ne in range(num_exp):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            HA_train, HA_test, yA_train, yA_test = train_test_split(XA, yA, test_size=per, random_state= 0)
            HB_train, HB_test, yB_train, yB_test = train_test_split(XB, yB, test_size=per, random_state= 0)

            fake_A = HA_train*np.random.binomial(size=HA_train.shape, n=1, p=1-.2)
            fake_B = HB_train*np.random.binomial(size=HB_train.shape, n=1, p=1-.2)

            for e in range(epochs):

                # ----------------------
                #  Run optimizers
                # ----------------------

                _ = sess.run(d_train_opt, feed_dict={DA_real: HA_train, DB_real: HB_train, input_zA: fake_A, input_zB: fake_B})
                _ = sess.run(g_train_opt, feed_dict={input_zA:fake_A , input_zB:fake_B})

                #----------------
                # Domain A and B
                #----------------

                train_loss_d = sess.run(d_loss, {DA_real: HA_train, DB_real: HB_train, input_zA: fake_A, input_zB: fake_B})
                train_loss_g = g_loss.eval({input_zA: fake_A , input_zB:fake_B})


                saver.save(sess, './checkpoints/generator.ckpt')


                    # Save losses to view after training
                losses.append((train_loss_d, train_loss_g))


            saver = tf.train.Saver(var_list=G_vars)
            with tf.Session() as sess:
                saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

                fake_A2 = HA_test*np.random.binomial(size=HA_test.shape, n=1, p=1-0)
                fake_B2 = HB_test*np.random.binomial(size=HB_test.shape, n=1, p=1-0)

                
                fake_A2B = sess.run(generatorA(input_zA, DA_size, n_units=gA_hidden_size, alpha=alpha, reuse=True),
                                   feed_dict={input_zA: fake_A2})

                fake_B2A = sess.run(generatorB(input_zB, DB_size, n_units=gB_hidden_size, alpha=alpha, reuse=True),
                                   feed_dict={input_zB: fake_B2})


                recov_A = sess.run(generatorA(input_zA, DA_size, n_units=gA_hidden_size, alpha=alpha, reuse=True),
                                   feed_dict={input_zA: fake_B2A})

                recov_B = sess.run(generatorB(input_zB, DB_size, n_units=gB_hidden_size, alpha=alpha, reuse=True),
                                   feed_dict={input_zB: fake_A2B})
                                   

#################################################################################################################
   #  Step 2: Feature Distribution Alignment #
#################################################################################################################

            w1 = KMM.iwe_kernel_mean_matching(HB_test, fake_B2A)
            w2 = KMM.iwe_kernel_mean_matching(HA_test, fake_A2B)


#################################################################################################################
   #  Step 3: Classification training and prediction #
#################################################################################################################
                                  
            parameters  = {'C':[.1,1],'gamma':[.1,1]}
            smv_dism_data1 = svm.SVC(kernel='rbf', probability = True)
            clf = GridSearchCV(smv_dism_data1, parameters)
            predicted_data1_svm = clf.fit(fake_A2B, yA_test, sample_weight= w2)
            pred_label_svm = predicted_data1_svm.predict(HB_test)
            
            rs2_svm = recall_score(yB_test, pred_label_svm, average='micro') 
            ps2_svm = precision_score(yB_test, pred_label_svm, average='micro') 
            
            print('F-score micro ... ')
            print(2*(rs2_svm*ps2_svm)/(rs2_svm+ps2_svm))
            
            rs_svm = recall_score(yB_test, pred_label_svm, average='macro') 
            ps_svm = precision_score(yB_test, pred_label_svm, average='macro') 
            
            print('F-score macro ... ')
            print(2*(rs_svm*ps_svm)/(rs_svm+ps_svm))
        
