# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 21:18:56 2016

@author: Tom
"""

import numpy as np
from numpy import linalg as LA
from scipy.stats import multivariate_normal


def normalize_data(X):
    """Normalize data by subtracting mean and divide by standard deviation"""
    mu = np.mean(X,axis=1)           # mean of each row
    sigma = np.std(X,axis=1,ddof=1)  # divide by n-ddof ~ delta degrees of freedom... not sure what this means
    normalized_data = (X - mu)/sigma # term-wise divide
    return normalized_data



def get_shuffled(X,D): 
    """ returns X_shuffled, D_shuffled by stacking X & D in a new matrix """    
    num_features = np.shape(X)[0]
    D_temp = D.T    
    X_temp = np.row_stack(( X, D_temp )).T
    np.random.shuffle(X_temp)
    X_shuf = np.matrix(X_temp[:,0:num_features]).T
    D_shuf = X_temp[:,3]
    
    return X_shuf, D_shuf



def shuffle_unison(X, D):
    """ shuffles in place by using same random key """
    rng_state = np.random.get_state()
    np.random.shuffle(X.T)
    np.random.set_state(rng_state)
    np.random.shuffle(D)



def shuffle_row_samples(X, D):
    """ shuffles in place by using same random key """
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(D)


def count_class_labels(D, label0=1, label1=-1):
    """ count how many samples from each class """
    """ returns num_pos, num_n  --> class0, class1 """
    
    num_p = 0
    num_n = 0
    
    for i in range(0,np.shape(D)[0]):
        if( D[i,0] == label0 ):  # positive class
            num_p = num_p + 1
        else:                    # negative class
            num_n = num_n + 1

    return num_p, num_n



def split_data(X, D, num_g1):
    return ( X[:,0:num_g1], D[0:num_g1, 0], X[:,num_g1:], D[num_g1:, 0])
    
    

def do_fld(X, D, get_bias = False):
    """ apply FLD and return the weight vector and bias """
    ''' split classes into their own matrices '''   
    Xp = np.matrix(np.zeros(np.shape(X)[0])).T # positive class
    Xn = np.matrix(np.zeros(np.shape(X)[0])).T # positive class    
    
    for i in range(0,np.shape(X)[1]):
        if( D[i,0] > 0 ):  # positive class
            Xp = np.column_stack(( Xp, X[:,i] ))
        else:               # negative class
            Xn = np.column_stack(( Xn, X[:,i] ))
            
    # end and remove the 0 columns
    Xp = np.delete(Xp, 0, 1)
    Xn = np.delete(Xn, 0, 1)
    
    ''' get matrices for FLD '''
    # get inner class means (mean of each row) and covariances
    mean_Xp = np.mean(Xp,axis=1) 
    mean_Xn = np.mean(Xn,axis=1) 
    COVp = np.cov(Xp)
    COVn = np.cov(Xn)

    ''' FLDA '''    
    # bewteen class covariance
    Sb = (mean_Xp - mean_Xn) * (mean_Xp - mean_Xn).T
    
    # within class sample covariance
    Sw = COVp + COVn
    
    #worked...
    m_p = np.shape(Xp)[1] #number samples
    m_n = np.shape(Xn)[1]
    m_tot = m_p + m_n
    Sw_slides = (m_p*COVp + m_n*COVn) / m_tot
    
    # FLD -> can I get the other solution for 2D projection?
    w_FLD = LA.inv( Sw )*(mean_Xp - mean_Xn)
    b = -(m_p*mean_Xp + m_n*mean_Xn).T * w_FLD / m_tot
    
    if(get_bias):
        return (w_FLD, b)
    else:
        return w_FLD
        


def do_LR(X, D, mu = 0.01, Beta = 1, iters = 1000, make_weight_plot = False):
    """ apply LR. return the weight vector and list of weights over time """    
    
    # set w_0
    w = np.matrix( np.zeros(np.shape(X)[0]) ).T
    i = 0  # current sample

    # plot weights
    w_list = np.matrix( np.zeros(shape=(np.shape(w)[0], iters)) )
    
    for j in range(0, iters):
        
        #overflow error is because the exp value is inf.. try normalizing data        
        
        #soft adjustments w/ probability sample is misclassified
        p_incorrect = 1 / ( 1 + np.exp(Beta*D[i,0]*w.T*X[:,i]) ) #err overflow
        w = w + mu*Beta*(X[:,i]*D[i,0])*p_incorrect
        
        #plotting each weight
        if(np.shape(X)[0] > 1):     # multi-dimensional
            if(LA.norm(w) > 0): # div 0 fix
                w_list[:,j] = w[:,0] / LA.norm(w)
        else:                       # one-dimensional
            w_list[:,j] = w[:,0]
            
        
        # goto next sample
        i = (i+1)%np.shape(X)[1]
        
    #end loop

    # return just w or w with the plot
    if(make_weight_plot):
        return (w,w_list)
    else:
        return w



def do_PLA(X, D, mu = 0.01, MAX_ITERS = 10000, make_weight_plot = False):
    """ Perceptron learning algorithm, also outputs the weights """    
    
    # set w_0
    w = np.matrix( np.zeros(np.shape(X)[0]) ).T

    # plot weights
    w_list = np.matrix( np.zeros(shape=(np.shape(w)[0], MAX_ITERS)) )    

    i_tot = 0 # total iterations to converge
    a_tot = 0 # total adjustments
    i = 0  # current sample
    c = 0  # correct counter
    
    while (c<np.shape(X)[1]):
        # while we havent classified all points correctly...
    
        if( w.T*X[:,i]*D[i,0] > 0 ):
            # correctly classified --> continue
            c += 1
        else:
            # incorrect --> modify weight vector
            w = w + mu * (X[:,i] * D[i,0])
            c = 0
            a_tot += 1
        #end
        
        #plotting each weight
        if(np.shape(X)[0] > 1):     # multi-dimensional
            if(LA.norm(w) > 0):
                w_list[:,i_tot] = w[:,0] / LA.norm(w)
        else:                       # one-dimensional
            w_list[:,i_tot] = w[:,0]
        
        # goto next sample
        i = (i+1)%np.shape(X)[1]
        i_tot += 1
        
        # watch dog        
        if(i_tot >= MAX_ITERS):
            print("\nERROR: NOT SEPERABLE -- iters:", str(MAX_ITERS), "\n")
            break
        
    #end loop
    
    if(make_weight_plot):
        return (w,w_list)
    else:
        return w



def do_naive_bayes_2_class(X, D, label0, label1):
    """ splits X based on labels, and trains two class classifier """    
    
    #split classes w/ np.extract (had to use compress)
    where_0 = np.array((D.T==label0))[0]
    where_1 = np.array((D.T==label1))[0]
    X0 = np.compress(where_0, X, axis=1)
    X1 = np.compress(where_1, X, axis=1)
    m0 = np.shape(X0)[1]
    m1 = np.shape(X1)[1]
    
    #get means and covariances (ML vars)
    mean_X0 = np.mean(X0,axis=1)
    mean_X1 = np.mean(X1,axis=1)
    cov_X0 = np.cov(X0)
    cov_X1 = np.cov(X1)
    
    # params
    return (mean_X0, mean_X1, cov_X0, cov_X1)



def test_naive_bayes_2_class(X, D, label0, label1, params, prior0=1, prior1=1):
    """ get error rate from given test data with labels """   

    # error counters
    err_0 = 0
    err_1 = 0

    # etract values from params/model
    mean_X0 = params[0]
    mean_X1 = params[1]
    cov_X0  = params[2]
    cov_X1  = params[3]
    
    # convert numpy types to python list types for multivariate fxn
    X_l = X.T.tolist()
    mean_X0_l = np.array(mean_X0.T).tolist()[0]
    mean_X1_l = np.array(mean_X1.T).tolist()[0]
    likelihood_0 = multivariate_normal.pdf(X_l, mean_X0_l, cov_X0)
    likelihood_1 = multivariate_normal.pdf(X_l, mean_X1_l, cov_X1)
    
    #convert back to numpy for priors
    post_0 = np.array(likelihood_0)*prior0
    post_1 = np.array(likelihood_1)*prior1
    
    #run through and get error rate
    for j in range(0, np.shape(X)[1]):
        if(D[j,0] == label0):
            if(post_0[j] < post_1[j]):
                err_0 += 1
        else:
            if(post_1[j] < post_0[j]):
                err_1 += 1
    
    return (err_0, err_1)



def test_classifier(X, D, w, get_margin = False, debug_print = False):
    """" see number of misclassified points of a linear classifier """    
    
    w_unit = w/LA.norm(w)
    min_p = None
    min_n = None
    
    num_p_misclfd = 0
    num_n_misclfd = 0
    
    for j in range(0, np.shape(X)[1]):
        
        dist = D[j,0]*(X[:,j].T*w_unit)
        
        if(dist > 0):                                       # correct
            if(debug_print):
                print(j, "class:", D[j,0], "\tdist:\t", dist)
        else:                                               # misclassified
            if(debug_print):            
                print(j, "class:", D[j,0], "\tdist:\t", dist, "MISCLSSFD!")
            
            if( D[j,0] > 0 ):   #class 1 error
                num_p_misclfd = num_p_misclfd + 1
            else:               #class 2 error
                num_n_misclfd = num_n_misclfd + 1
                
            continue  # misclassified points dont affect margin
        #endif
            
        
        if( D[j,0] > 0 ):       #class (1)
            if(min_p is None):
                min_p = dist
            else:
                if(dist < min_p):
                    min_p = dist
        else:                   #class (-1)
            if(min_n is None):
                min_n = dist
            else:
                if(dist < min_n):
                    min_n = dist
        #end
    #end loop
    
    if(get_margin):
        return (num_p_misclfd, num_n_misclfd, min_p, min_n)
    else:
        return (num_p_misclfd, num_n_misclfd)



def test_SVM_model(X_array, D_array, model):
    """ Get the error rate of an SVM """
    
    pred = model.predict( X_array )
    
    num_p_misclfd = 0
    num_n_misclfd = 0   
    
    for j in range(0, np.shape(pred)[0]):        
        if( pred[j] != D_array[j] ):
            if( D_array[j] > 0 ):   # class 1 error
                num_p_misclfd = num_p_misclfd + 1
            else:                   # class 2 error
                num_n_misclfd = num_n_misclfd + 1
    
    return (num_p_misclfd, num_n_misclfd)


def test_SVM_model_err(X_array, D_array, model):
    """ Get the error rate of an SVM """
    
    pred = model.predict( X_array )
    
    num_misclfd = 0
    for j in range(0, np.shape(pred)[0]):        
        if( pred[j] != D_array[j] ):
            if( D_array[j] > 0 ):   # class error
                num_misclfd = num_misclfd + 1
    
    return num_misclfd


def test_SVM_reg_model(X_array, D_array, model):
    """ Get the mean square error of SVM regression """
    
    pred = model.predict( X_array ) # predicted outputs, x*w.T
    m = np.shape(pred)[0]
    mse = (1/m) * np.sum( np.square(pred - D_array) )
    return mse



def test_SVM_reg_model_with_w(X, D, w, bias):
    """ Get the mse of SVM regression with weight vector """
    
    pred = X.T * w + bias
    m = np.shape(pred)[0]
    mse = (1/m) * np.sum( np.square(pred - D) )
    return mse
    


def test_LS_SVM_reg(X_test, D_test, X_train, alphas, bias):
    """ 
    Get the mse of LS SVM regression with dual solution 
    NOTE: D_test must line up with X_test (same dimension/order)
    """    
    
    # all samples in the training set are support vectors!
    pred = np.sum(np.multiply(X_train.T*X_test, alphas), axis=0) + bias
    pred = pred.T
    m = np.shape(pred)[0]
    mse = (1/m) * np.sum( np.square(pred - D_test) )
    return mse



def test_LS_SVM_poly_reg(X_test, D_test, X_train, alphas, bias, c=1):
    """ 
    Get the mse of LS SVM regression with dual solution 
    NOTE: D_test must line up with X_test (same dimension/order)
    """    
    
    # poly kernel
    K = np.square(X_train.T*X_test + c)
    
    # all samples in the training set are support vectors!
    pred = np.sum(np.multiply(K, alphas), axis=0) + bias
    pred = pred.T
    m = np.shape(pred)[0]
    mse = (1/m) * np.sum( np.square(pred - D_test) )
    return mse
    
    

def block_matrix_inv(A, B, C, D):
    Ai = LA.inv(A)
    Di = LA.inv(D)
    
    A_out = LA.inv( (A - B*Di*C) )
    B_out = -Ai*B*LA.inv(D-C*Ai*B)
    C_out = -Di*C*LA.inv(A-B*Di*C)
    D_out = LA.inv(D-C*Ai*B)
    
    return (A_out, B_out, C_out, D_out)
