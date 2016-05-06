import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from scipy.stats import multivariate_normal
from sklearn import svm, linear_model
from util import tomML as ml
import time

# vars
c_data_full = np.array([])    # full user data
c_datas = []                  # data from each class
c_classes = []                # class for each datas[i]
c_classifier = 0              # type of classifier being used
c_params = []                 # classifier params (holding each machine)
                                  # nb      - [[mean, cov], ...]
                                  # svm     - [[svm_model], ...]
                                  # svm_rbf - [[svm_model], ...]
c_machine_params = []         # machine learning constants (found by validation)
                                  # nb      - [priors, ...]
                                  # svm     - [C] 
                                  # svm_rbf - [[C, u]]
c_normalize = False           # normalize data option
c_mean_full = None            # ""
c_std_full = None             # ""

# enum for choosing different classifiers
c_naive_bayes = 0
c_svm_rbf     = 1
c_lr          = 2


# for performing scalable tests
c_data_test = np.array([])    # test data (only intended to be local)


def get_data_from(full_data_path, reserve_test_set = False, test_set_size = 15, class_limit=None, limit_before=False):
    global c_data_full, c_datas, c_classes
    global c_data_test
    global c_mean_full, c_std_full
    
    # extract data from file
    c_data_full = np.genfromtxt(full_data_path, delimiter ='\t')
    
    # remove classes that are over the cap if desired (before)
    if(class_limit != None and limit_before):
        c_data_full = c_data_full[ c_data_full[:,0] < class_limit ]
    
    # remove some of the data for testing (not used in the actual app)
    if(reserve_test_set):
        np.random.shuffle(c_data_full)
        c_data_test = c_data_full[0:test_set_size]
        c_data_full = c_data_full[test_set_size::]
    
    # data mean and variance
    X = np.matrix(c_data_full[:, 1::]).T  # data as columns
    c_mean_full = np.mean(X,axis=1)       # mean of each row
    c_std_full = np.std(X,axis=1,ddof=1)  # divide by n-ddof
    
    # get class labels and fill in datas
    c_datas = []    
    c_classes = []    
    for i in range(0, np.shape(c_data_full)[0]):
        c = c_data_full[i, 0]                  # class of current sample
        if(not c in c_classes):
            # new class?
            c_classes.append(c)                    # place it in c_classes list
            start = np.matrix(c_data_full[i, 1::]) # fix to keep sample as a row
            c_datas.append(np.array(start))        # start a new list inside of datas
        else:
            # existing class
            c_datas[c_classes.index(c)] = np.row_stack((
                                              c_datas[c_classes.index(c)],
                                              c_data_full[i, 1::]))
            continue
    # endfor
    
    
    # limit num of classes after data is randomized
    if(class_limit != None and not limit_before):
        if(len(c_classes) > class_limit):
            
            # limit the classes
            c_classes = c_classes[0:class_limit]
            c_datas = c_datas[0:class_limit]
    
            # prune the non-class types out of the testing data
            belongs = np.zeros(np.shape(c_data_test)[0])
            for i in range(0, len(c_classes)):
                belongs += (c_data_test[:,0] == c_classes[i]).astype(int)
            c_data_test = c_data_test[belongs.astype(bool)]
            1 == 1
    
    

# for using normalization (NOTE: should only be called once after getting data)
def use_normalization_and_normalize_training_data():
    global c_datas
    global c_mean_full, c_std_full
    global c_normalize

    # use boolean to not normalize multiple times...
    c_normalize = True
    
    mu = c_mean_full           # mean of each row
    sigma = c_std_full    
    
    # normalize all datas[i]
    for i in range(0, len(c_datas)):
        datas_i_m = np.matrix(c_datas[i]).T
        c_datas[i] = np.array( ((datas_i_m - mu)/sigma).T ) # term-wise divide



def set_classifier(classifier_type):
    global c_classifier    
    c_classifier = classifier_type
    

def train(machine_params=None, use_validation=False, val_range=[(-5,5,10), (-5, 5, 10)], folds=5):
    """ train the ML algo. if use_valiation, then machine_params dont matter """
    
    global c_datas, c_classes
    global c_naive_bayes, c_svm, c_lr    
    global c_classifier
    global c_params, c_machine_params
    
    c_params = []
    
    # Naive Bayes
    if(c_classifier == c_naive_bayes):
        
        # if no machine_params, assume equal priors...
        if(machine_params == None):
            c_machine_params = [1]*len(c_datas)
        
        # gets means and covariances from each class (params for each machine)
        for i in range(0, len(c_datas)):
        
            # make data matrix (nxm)
            X_i = np.matrix(c_datas[i]).T
            
            # get means and covs
            mean_i = np.mean(X_i, axis=1)
            cov_i = None            
            
            # if only a single sample... use identity matrix cov?
            if(np.shape(X_i)[1] <= 1):
                cov_i = np.eye(np.shape(X_i)[0])
            else:
                cov_i  = np.cov(X_i)
            
            # populate the c_params (will be used for classification)
            c_params.append([mean_i, cov_i])
            
        #endfor
        
        # do cross-validation to find priors? Not necessary...
        
    
    # SVM RBF (one-vs-rest multi-class SVM)
    if(c_classifier == c_svm_rbf):
        
        ''' join data together '''
        X = np.zeros(( 1, np.shape(c_datas[0])[1] ))
        d = np.zeros(1)
        for i in range(0, len(c_datas)):
            X = np.row_stack((X, c_datas[i]))
            d = np.append( d, np.full(np.shape(c_datas[i])[0], c_classes[i]) )
        X = X[1::]
        d = d[1::]
        
        # size of training set        
        m_tr = np.shape(X)[0]
        
        ''' not using cross validation? '''
        if(not use_validation):
            
            # create model and fit
            model = svm.SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
                            decision_function_shape='ovr', gamma='auto', 
                            kernel='rbf', max_iter=-1, probability=False,  
                            random_state=None, shrinking=True, tol=0.001, verbose=False)
            model.fit(X, d)
            # send off the model
            c_params.append([model])
            return
        
        
        ''' otherwise do cross validation '''        

        # validation test range
        C_list = np.logspace(val_range[0][0], val_range[0][1], val_range[0][2], base = 2)          
        G_list = np.logspace(val_range[1][0], val_range[1][1], val_range[1][2], base = 2)            
        
        # shuffle
        ml.shuffle_row_samples(X, d)
        
        # vars to save best validation model
        mse_vl_min = None  
        C_best = None
        G_best = None        
        
        # iterate over C values
        for j in range(0, C_list.size):
            
            # iterate over G values (width of gausian)
            for jj in range(0, G_list.size):
            
                # get parameters we are testing
                C = C_list[j]
                G = G_list[jj]
                
                # reset err for C value before each run of cross validation
                err_vl_ave = 0
                
                # start k-fold cross val, (i_start and i_end ~ training data)
                X_tr_tr = np.zeros(1)
                D_tr_tr = np.zeros(1)
                X_tr_vl = np.zeros(1)
                D_tr_vl = np.zeros(1)
                i_start = 0
                i_end = m_tr - int(m_tr/folds)
                
                # each k-fold cross validation
                for k in range(0, folds):
                    
                    #reassign training and validation sets
                    if (i_start < i_end):
                        X_tr_tr = X[i_start:i_end,:]
                        D_tr_tr = d[i_start:i_end]
                        X_tr_vl = X[i_end::, :]
                        D_tr_vl = d[i_end::]
                    else:
                        if(i_end != 0):
                            X_tr_tr = np.row_stack((X[i_start::, :], X[0:i_end, :]))
                            D_tr_tr = np.append(d[i_start::], d[0:i_end])
                        else:
                            X_tr_tr = X[i_start::, :]
                            D_tr_tr = d[i_start::]
                        X_tr_vl = X[i_end:i_start, :]
                        D_tr_vl = d[i_end:i_start]
                    
                    
                    # create model
                    model_vl = svm.SVC(C=C, cache_size=200, class_weight=None, coef0=0.0,
                                        decision_function_shape='ovr', gamma=G, 
                                        kernel='rbf', max_iter=-1, probability=False,  
                                        random_state=None, shrinking=True, tol=0.001, verbose=False)
                                
                    # fit to training data
                    model_vl.fit(X_tr_tr, D_tr_tr)
                    
                    # test classifier on validation set
                    mse_tr_vl = ml.test_SVM_model_err(X_tr_vl, D_tr_vl, model_vl)
                    
                    # add to error terms
                    err_vl_ave += mse_tr_vl/folds
                    
                    # increment to next fold
                    i_start = int(i_start + m_tr/folds) % m_tr
                    i_end = int(i_end + m_tr/folds) % m_tr
                    
                #endfor, cross validation (got average validation err)
    
                # check if the resulting mse_vl is less than the best
                if(not (mse_vl_min is None)):
                    if(err_vl_ave <= mse_vl_min):
                        mse_vl_min = err_vl_ave
                        C_best = C
                        G_best = G
                else:
                    mse_vl_min = err_vl_ave
                    C_best = C
                    G_best = G            

            #endfor, G values

        #endfor, C values
        
        # C best and G best are now found
        print("C best: " + str(C_best) + "  G best: " + str(G_best))
        
        # create model and fit to data
        model = svm.SVC(C=C_best, cache_size=200, class_weight=None, coef0=0.0,
                        decision_function_shape='ovr', gamma=G_best, 
                        kernel='rbf', max_iter=-1, probability=False,  
                        random_state=None, shrinking=True, tol=0.001, verbose=False)
        model.fit(X, d)
        
        # send off the model
        c_params.append([model])
        
    # logistic regression
    if(c_classifier == c_lr):
        
        ''' join data together '''
        X = np.zeros(( 1, np.shape(c_datas[0])[1] ))
        d = np.zeros(1)
        for i in range(0, len(c_datas)):
            X = np.row_stack((X, c_datas[i]))
            d = np.append( d, np.full(np.shape(c_datas[i])[0], c_classes[i]) )
        X = X[1::]
        d = d[1::]
        
        # size of training set        
        m_tr = np.shape(X)[0]
        
        ''' not using cross validation? '''
        if(not use_validation):
            
            # create model and fit
            model = linear_model.LogisticRegression(C=1000)
            model.fit(X, d)
            # send off the model
            c_params.append([model])
            return
        
        
        ''' otherwise do cross validation '''        

        # validation test range
        C_list = np.logspace(val_range[0][0], val_range[0][1], val_range[0][2], base = 2)           
        
        # shuffle
        ml.shuffle_row_samples(X, d)
        
        # vars to save best validation model
        mse_vl_min = None  
        C_best = None
        
        # iterate over C values
        for j in range(0, C_list.size):
        
            # get parameters we are testing
            C = C_list[j]
            
            # reset err for C value before each run of cross validation
            err_vl_ave = 0
            
            # start k-fold cross val, (i_start and i_end ~ training data)
            X_tr_tr = np.zeros(1)
            D_tr_tr = np.zeros(1)
            X_tr_vl = np.zeros(1)
            D_tr_vl = np.zeros(1)
            i_start = 0
            i_end = m_tr - int(m_tr/folds)
            
            # each k-fold cross validation
            for k in range(0, folds):
                
                #reassign training and validation sets
                if (i_start < i_end):
                    X_tr_tr = X[i_start:i_end,:]
                    D_tr_tr = d[i_start:i_end]
                    X_tr_vl = X[i_end::, :]
                    D_tr_vl = d[i_end::]
                else:
                    if(i_end != 0):
                        X_tr_tr = np.row_stack((X[i_start::, :], X[0:i_end, :]))
                        D_tr_tr = np.append(d[i_start::], d[0:i_end])
                    else:
                        X_tr_tr = X[i_start::, :]
                        D_tr_tr = d[i_start::]
                    X_tr_vl = X[i_end:i_start, :]
                    D_tr_vl = d[i_end:i_start]
                
                
                # create model
                model_vl = linear_model.LogisticRegression(C=C)
                
                # fit to training data
                model_vl.fit(X_tr_tr, D_tr_tr)
                
                # test classifier on validation set
                mse_tr_vl = ml.test_SVM_model_err(X_tr_vl, D_tr_vl, model_vl)
                
                # add to error terms
                err_vl_ave += mse_tr_vl/folds
                
                # increment to next fold
                i_start = int(i_start + m_tr/folds) % m_tr
                i_end = int(i_end + m_tr/folds) % m_tr
                
            #endfor, cross validation (got average validation err)

            # check if the resulting mse_vl is less than the best
            if(not (mse_vl_min is None)):
                if(err_vl_ave <= mse_vl_min):
                    mse_vl_min = err_vl_ave
                    C_best = C
            else:
                mse_vl_min = err_vl_ave
                C_best = C

        #endfor, C values
        
        # C best is now found
        print("C best: " + str(C_best))
        
        # create model and fit to data
        model = linear_model.LogisticRegression(C=C_best)
        model.fit(X, d)
        
        # send off the model
        c_params.append([model])



def classify(X_test_data, print_output = False):
    """ classify new data, X_test is mxn array (row-samples) """

    global c_classes    
    global c_params, c_machine_params    
    global c_normalize
    global c_mean_full, c_std_full
    
    X_test = np.matrix(X_test_data)                 # fix: ensure samples are rows
    X_test = np.array(X_test)
    
    # normalize if normalization is on
    if(c_normalize == True):
        mu = c_mean_full
        sigma = c_std_full
        X_T = np.matrix(X_test).T                   # convert to matrix to do math
        X_test = np.array( ((X_T - mu)/sigma).T )   # term-wise divide
    
    # vars for classification
    d_test = np.zeros(np.shape(X_test)[0])     # class labels
    prob_i = np.zeros(np.shape(X_test)[0])     # stores max probabilities
    
    
    if(c_classifier == c_naive_bayes):

        # get probabilities of all samples belonging to class i
        for i in range(0, len(c_classes)):
            
            # get priors from the machine params
            prior_i = c_machine_params[i]
            
            # get probability for each sample
            X_l = X_test.tolist()
            mean_i = np.array(c_params[i][0].T).tolist()[0]
            cov_i = c_params[i][1]
            cov_i = np.diag( np.diag(cov_i) ) # try diag (singular matrix fix)
            likelihood_i = multivariate_normal.pdf(X_l, mean_i, cov_i)
            
            # single element fix...
            if(np.array(likelihood_i).size == 1):
                likelihood_i = [likelihood_i]
            
            posterior_i = np.array(likelihood_i)*prior_i
            
            # set d to this class if posterior is higher
            d_test[ posterior_i > prob_i ] = c_classes[i]
            
            # set new probs for future classes to compete with
            prob_i[ posterior_i > prob_i ]= posterior_i[ posterior_i > prob_i ]

        #endfor
    
    
    if(c_classifier == c_svm_rbf):
        
        # grad model from params
        model = c_params[0][0]
        
        # classify and also get distances from hyperplane
        d_test = model.predict(X_test)
        dec_test = model.decision_function(X_test)
        num_supp = model.n_support_

        if(print_output):
            print(d_test)
    
    
    if(c_classifier == c_lr):
        
        # grad model from params
        model = c_params[0][0]
        
        # classify and also get distances from hyperplane
        d_test = model.predict(X_test)
        dec_test = model.decision_function(X_test)

        if(print_output):
            print(d_test)
    
        
    return d_test.astype(int).tolist()



######## module test #########

def main():
    
    my_full_path = "./data/user/tom.csv"
    
    num_algos = 3
    max_classes = 5
    num_epochs = 20
    
    err_tr_arr = np.zeros((num_algos, max_classes-1))
    err_ts_arr = np.zeros((num_algos, max_classes-1))
    time_arr = np.zeros(num_algos)
    
    conf_matrix = np.zeros((max_classes, max_classes))
    
    for c_num in range(0, num_algos):

        for labels in range(1, max_classes):
            
            err_tr_ave = 0
            err_ts_ave = 0
            
            for i in range(0, num_epochs):
                
                # set classifier
                set_classifier(c_num)
                
                # get data (will shuffle since reserving data)
                get_data_from(my_full_path, True, 75, labels+1, True)
                
                # normalize data (only once after getting data)
                use_normalization_and_normalize_training_data()
                    
                # train (machine params, valiation...)
                #train(None, True, [(-3, 10, 10), (-3, 10, 10)], 5) #svm
                train() #logistic regression
                
                #run on training data
                my_training_data = c_data_full[:, 1::]
                my_training_data_d = c_data_full[:,0]    
                y = classify(my_training_data)
                
                y_arr = np.array(y)
                d_arr = my_training_data_d.astype(int)
                err = np.sum( (y_arr != d_arr).astype(int) )
                percent_err = err / np.size(d_arr)
                err_tr_ave += percent_err / num_epochs
                
                # test on testing data
                my_test_data = c_data_test[:, 1::]
                my_test_data_d = c_data_test[:,0]    
                y = classify(my_test_data)
                
                y_arr = np.array(y)
                d_test_arr = my_test_data_d.astype(int)
                err = np.sum( (y_arr != d_test_arr).astype(int) )
                percent_err = err / np.size(d_test_arr)
                err_ts_ave += percent_err / num_epochs
                
                # get confusion matrix on last iteration
                if(labels == max_classes-1):
                    conf_matrix[y_arr, d_test_arr] += 1
                
            #end epochs for stats
            err_ts_arr[c_num][labels-1] = err_ts_ave*100
            err_tr_arr[c_num][labels-1] = err_tr_ave*100
            
        #end label count
    
    #end algos
    
    
    # test runtimes
    for c_num in range(0, num_algos):
        
        # set classifier
        set_classifier(c_num)
        
        # get data (will shuffle since reserving data)
        get_data_from(my_full_path)
        
        # normalize data (only once after getting data)
        use_normalization_and_normalize_training_data()
            
        # train (machine params, valiation...)
        train() #logistic regression
        
        start_time = time.time()
        for i in range(0, 100):
            y = classify(my_test_data)
        end_time = time.time()
    
        time_arr[c_num] = end_time - start_time
    
    
    # print
    print("Machine: " + str(c_classifier))
    print("Classes: " + str(len(c_classes)) + " --> " + str(c_classes))
    print("Err: ", err,  "/", np.size(d_test_arr))
    print("% err: ", err / np.size(d_test_arr))
    
    print("Mistakes: ", y_arr[y_arr != d_test_arr])
    print("Actual  : ", d_test_arr[y_arr != d_test_arr])
    
    print("Conf Matrix: ")
    print(conf_matrix)
        
    print("Classification Times")
    print(time_arr)        
        
    #plot
    plt.figure(1)
    for j in range(0, np.shape(err_ts_arr)[0]):
        plt.plot(np.arange(2,6), err_ts_arr[j][:], label="ts err"+str(j))
    plt.legend(loc=2)
    plt.ylabel("Testing error rate (%)")
    plt.xlabel("Num Classes (C)")
    
    plt.figure(2)
    for j in range(0, np.shape(err_ts_arr)[0]):
        plt.plot(np.arange(2,6), err_tr_arr[j][:], label="tr err"+str(j))
    plt.legend(loc=2)
    plt.ylabel("Training error rate (%)")
    plt.xlabel("Num Classes (C)")
    
    
    plt.show()
    print("HELLO")

# main()


'''
tr_err[algo][#classes]  ->
ts_err[algo][#classes]  ->   Plot all of these
tr_t[algo][#classes]    ->
cl_t[algo][#classes]    ->

with & without cross validation

for(each classifier)

   for(using 2 to 5 class labels)

        ave_tr_err
        ave_ts_err
        for(30 epochs)
            shuffle and retrain

        plot err:
           - training 
           - testing
        
        --> print worst confusion
        
        etract training time
        
        extract classification time        
    
'''


'''
::vars::
    full_data:Matrix    # the matrix of all data
    
    class_data[]:Matrix #data on each class 0-numClasses
    
    classifier:int --> 
        0 - naive bayes
        1 - svm
        2 - lr

::fxns::
    get_data(full_data_path)
        set full_data    
    
    set_classifier_type(type)
    
    train() //uses data from data_file
      --> creates the model for the training and saves in in c_params
    
    classify(X_test)
      --> takes models from training and assigns class labels
        
    
idea:   somehow restrict train() to use a subset of the full_data
        then, give classify the other remaining set of data to test
        
        advantage: 
            fxns train and test abstract the multiple-machines
            
        ---> solution: make an option for get data to store 
        
        
'''


'''
my_outer_test_path = "./data/blown_beat_data.csv"

# test on outer data
my_test_data = np.genfromtxt(my_outer_test_path, delimiter ='\t')
my_test_data_d = my_test_data[:,0]
my_test_data = my_test_data[:, 1::]    
'''
    
