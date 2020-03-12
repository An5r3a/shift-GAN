import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score
from sklearn.svm import SVC
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 

import scipy.stats as st
from scipy.spatial.distance import cdist
import sklearn as sk
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

def generatorA(z, out_dim, n_units, reuse=False,  alpha=0.1):
    ''' Build the generator network.
    
        Arguments
        ---------
        z : Input tensor for the generator
        out_dim : Shape of the generator output
        n_units : Number of units in hidden layer
        reuse : Reuse the variables with tf.variable_scope
        alpha : leak parameter for leaky ReLU
        
        Returns
        -------
        out: tanh function
    '''
    with tf.variable_scope('generatorA', reuse=reuse):
        # Hidden layer 1
        h1 = tf.layers.dense(z, n_units, activation=None)
        # Leaky ReLU
        h1 =  tf.nn.relu(h1)  - alpha * tf.nn.relu(-h1)
       
        # Logits and tanh output
        logits = tf.layers.dense(h1, out_dim, activation=None)
        out = tf.sigmoid(logits)
        
        return out
        
def generatorB(z, out_dim, n_units, reuse=False,  alpha=0.1):
    ''' Build the generator network.
    
        Arguments
        ---------
        z : Input tensor for the generator
        out_dim : Shape of the generator output
        n_units : Number of units in hidden layer
        reuse : Reuse the variables with tf.variable_scope
        alpha : leak parameter for leaky ReLU
        
        Returns
        -------
        out: tanh function
    '''
    with tf.variable_scope('generatorB', reuse=reuse):
        # Hidden layer 1
        h1 = tf.layers.dense(z, n_units, activation=None)
        # Leaky ReLU
        h1 =  tf.nn.relu(h1)  - alpha * tf.nn.relu(-h1)
       
        # Logits and tanh output
        logits = tf.layers.dense(h1, out_dim, activation=None)
        out = tf.sigmoid(logits)
        
        return out      
        
        
def discriminatorA(x, n_units, reuse=False, alpha=0.1):
    ''' Build the discriminator network.
    
        Arguments
        ---------
        x : Input tensor for the discriminator
        n_units: Number of units in hidden layer
        reuse : Reuse the variables with tf.variable_scope
        alpha : leak parameter for leaky ReLU
        
        Returns
        -------
        out, logits: 
    '''
    with tf.variable_scope('discriminatorA', reuse=reuse):
        # Hidden layer 1
        h1 = tf.layers.dense(x, n_units, activation=None)
        # Leaky ReLU
        h1 = tf.maximum(alpha * h1, h1)
        
        # Hidden layer 2
        h2 = tf.layers.dense(h1, n_units, activation=None)
        # Leaky ReLU
        h2 = tf.maximum(alpha * h2, h2)
        
        logits = tf.layers.dense(h2, 1, activation=None)
        out = tf.sigmoid(logits)
        
        return out, logits 
        
        
def discriminatorB(x, n_units, reuse=False, alpha=0.1):
    ''' Build the discriminator network.
    
        Arguments
        ---------
        x : Input tensor for the discriminator
        n_units: Number of units in hidden layer
        reuse : Reuse the variables with tf.variable_scope
        alpha : leak parameter for leaky ReLU
        
        Returns
        -------
        out, logits: 
    '''
    with tf.variable_scope('discriminatorB', reuse=reuse):
        # Hidden layer 1
        h1 = tf.layers.dense(x, n_units, activation=None)
        # Leaky ReLU
        h1 = tf.maximum(alpha * h1, h1)
        
        # Hidden layer 2
        h2 = tf.layers.dense(h1, n_units, activation=None)
        # Leaky ReLU
        h2 = tf.maximum(alpha * h2, h2)
        
        logits = tf.layers.dense(h2, 1, activation=None)
        out = tf.sigmoid(logits)
        
        return out, logits 
