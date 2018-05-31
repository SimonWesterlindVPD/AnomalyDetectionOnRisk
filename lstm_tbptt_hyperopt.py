import matplotlib.pyplot as plt
import numpy as np
import time
import re
import os
import csv
import tensorflow as tf
import datetime
from scipy.stats import norm
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import CuDNNLSTM
from keras.models import Sequential
from keras.optimizers import Adam
from numpy import arange, sin, pi, random, genfromtxt
from pandas import concat
from pandas import DataFrame




#The LSTM class is capable of fetching data, performing anomaly detection on said data, calculates the score and plot the result.
#The LSTM is run sequentially and is updated using Truncated Backpropagation Through Time. Searches for hyperparameters by performing a gridsearch on the first serie in the dataset.

class lstm_tbptt(): 
    


    def hyperopt(self, data):

        print("Hyperopting!!")
        λ = 2
        best_mse = 99999999
        best_mse_with_regu = 99999999
        n_epoch = 1
        n_neurons = 10
        timesteps = 30
        features = 1
        learning_rate_spectrum = [0.005, 0.01, 0.02]
        neuron_setup_spectrum = [[30, 30], [15, 15, 15, 20], [50], [50, 50], [10, 10, 10], [15,15,15,15]]
        for learning_rate in learning_rate_spectrum:
            for neuron_setup in neuron_setup_spectrum:
                tot_mse = 0
                tot_regu = 0
                tot_mse_with_regu = 0
                preds = [0]
                actuals =[0]
                model = Sequential()
                if(len(neuron_setup) == 1):
                    model.add(CuDNNLSTM(neuron_setup[0], batch_input_shape=(1, timesteps, features), stateful=True))
                else:
                    model.add(CuDNNLSTM(neuron_setup[0], batch_input_shape=(1, timesteps, features), return_sequences=True, stateful=True))
                for layer in range(1, len(neuron_setup)):
                    model.add(Dropout(0.2))
                    if layer == (len(neuron_setup)-1):
                        model.add(CuDNNLSTM(neuron_setup[layer], stateful=True))
                    else:
                        model.add(CuDNNLSTM(neuron_setup[layer], return_sequences=True, stateful=True))

                model.add(Dense(1))

                opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
                model.compile(loss='mean_squared_error', optimizer=opt)
                for t in range(timesteps + 1, int(len(data)*0.1*0.8)):
                    # Assigns what data is available 
                    X_train = data[t-1-timesteps:t-1]
                    y_train = data[t]
                    X_train = X_train.reshape(1, timesteps, 1)
                    y_train = y_train.reshape(1, 1)

                    
                    # fit network
                    for i in range(n_epoch):
                        model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=0, shuffle=False)
                    
                for t in range(int(len(data)*0.1*0.8) + timesteps + 1, int(len(data)*0.1)):
                    X_val = data[t-1-timesteps:t-1]
                    y_val = data[t]
                    X_val = X_val.reshape(1, timesteps, 1)
                    y_val = y_val.reshape(1, 1)



                    y_hat = model.predict(X_val, batch_size=1)
                    the_actual = y_val[0][0]
                    the_predicted = y_hat[0][0]

                    preds.append(the_predicted)
                    actuals.append(the_actual)
                    tot_mse += (the_actual-the_predicted)**2
                    if len(preds) > 4:
                        tot_regu += (preds[-1]-preds[-2])**2 + (preds[-1]-preds[-3])**2 + (preds[-1]-preds[-4])**2 + (preds[-1]-preds[-5])**2
                    #print('>Expected=%.1f, Predicted=%.1f' % (the_actual, the_predicted))


                    for i in range(n_epoch):
                        model.fit(X_val, y_val, epochs=1, batch_size=1, verbose=0, shuffle=False)


                tot_mse_with_regu = tot_mse + λ*tot_regu
                #print(learning_rate, neuron_setup)
                #print("mse and regu: " + str(tot_mse_with_regu))
                if tot_mse_with_regu < best_mse_with_regu:
                    #print("Found new best model!")
                    best_model = model
                    best_mse = tot_mse
                    best_mse_with_regu = tot_mse_with_regu
                #plt.plot(actuals[1:])
                #plt.plot(preds[1:])
                #plt.show()

                #print(" ")
        model.reset_states()
        print("Hyper-opt finished")
        print(best_model)
        return best_model


    def LSTM(self, data, model):





        preds = np.zeros(len(data))
        actuals = np.zeros(len(data))
        n_epoch = 3
        n_neurons = 10
        timesteps = 30
        features = 1



        for t in range(timesteps + 1, len(data)):
            # Assigns what data is available 
            X = data[t-1-timesteps:t-1]
            y = data[t]
            X = X.reshape(1, timesteps, 1)
            y = y.reshape(1, 1)

            # fit network
            for i in range(n_epoch):
                model.fit(X, y, epochs=1, batch_size=1, verbose=0, shuffle=False)
                


            testX, testy = data[t-timesteps-1:t-1], data[t]
            testX = testX.reshape(1, timesteps, 1)
            yhat = model.predict(testX, batch_size=1)
            the_actual = testy
            the_predicted = yhat
            #print('>Expected=%.1f, Predicted=%.1f' % (the_actual, the_predicted))
            preds[t] = yhat[0]
            actuals[t] = testy




        model.reset_states()
        print(".", end='', flush=True)
        return preds


    def runAnomalyDetection(self, D, D_truth, D_unpolluted, dataType, dataSize):

        tf.logging.set_verbosity(tf.logging.ERROR)
        ϵ = 0.0001
        data = D

        FP = 0.
        FN = 0.
        TP = 0.
        TN = 0.

        FN_ALO = 0.
        TP_ALO = 0.
        FN_LS = 0.
        TP_LS = 0.
        FN_TC = 0.
        TP_TC = 0.
        FN_LTO = 0.
        TP_LTO = 0.
        FN_SALO = 0.
        TP_SALO = 0.

        reaction_counter = 0.
        tot_reaction = 0.

        print("Running LSTM-AD")
        guessed_anomalies = np.zeros((len(D), len(D[0])))
        all_preds = np.zeros((len(D), len(D[0])))
        all_residuals = np.zeros((len(D), len(D[0])))
        all_likelihoods = np.zeros((len(D), len(D[0])))

        #Get model
        model = self.hyperopt(D[0])
        for i in range(len(D)):
            preds = self.LSTM(D[i], model)
            all_preds[i] = preds




            #Make guesses
            for t in range(int(0.1*len(D[i])), len(D[i])):

                """
                Calculates the Anomaly Likelihood from the anomaly scores.
                The anomly score for the LSTM is concidered to be the absolute value of the residuals.
                W (a.k.a l) is how far back we look to determine what the distribution of anomaly scores should be.
                W_prim is how far back we look to detemine the current distribution of anomaly scores.
                Sources:
                Page 139 in https://ac.els-cdn.com/S0925231217309864/1-s2.0-S0925231217309864-main.pdf?_tid=56c94234-44b6-4223-afb9-3766ea4e5e41&acdnat=1523276715_025b50926fdc50d0e9085dff836b5c29
                Page 90 in https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2015-56.pdf

                """
                w = 50
                w_prim_1 = 1
                w_prim_2 = 5


                μ = 0
                for index_past in range(max(0, t-w), t):
                    s = abs(D[i, index_past] - preds[index_past]) #Absolute residual.
                    μ += s
                μ /= w

                σ = 0
                for index_past in range(max(0, t-w), t):
                    s = abs(D[i, index_past] - preds[index_past]) #Absolute residual.
                    σ += (s-μ)**2
                σ /= (w-1)
                σ = np.sqrt(σ)
                
                μ_tilde_1 = 0
                for index_past in range(max(0, t-w_prim_1), t):
                    s = abs(D[i, index_past] - preds[index_past]) #Absolute residual.
                    μ_tilde_1 += s
                μ_tilde_1 /= w_prim_1

                
                μ_tilde_2 = 0
                for index_past in range(max(0, t-w_prim_2), t):
                    s = abs(D[i, index_past] - preds[index_past]) #Absolute residual.
                    μ_tilde_2 += s
                μ_tilde_2 /= w_prim_2

                Q_1 = norm.sf((μ_tilde_1 - μ) / σ)
                L_1 = 1 - Q_1
                Q_2 = norm.sf((μ_tilde_2 - μ) / σ)
                L_2 = 1 - Q_2
                all_likelihoods[i][t] = L_1
                all_residuals[i][t] = D[i][t] - preds[i]

                if  L_1 >= 1 - ϵ or L_2 >= 1 - ϵ:
                    guessed_anomalies[i][t] = 1 #This is an anomaly guess

            #Evaluate guesses
            in_anomaly_window = False
            anomaly_window_type = 0
            has_flagged_anomaly_window = False
            for t in range(1, len(D[i])):

                #Exiting an anomaly window. Check if it has been flagged.
                if D_truth[i][t] == 0 and in_anomaly_window:
                    in_anomaly_window = False

                    if not has_flagged_anomaly_window: 
                        FN += 1 #Failed to flag the entire anomaly window.
                        if int(anomaly_window_type) == 1:
                            FN_ALO += 1 #Failed to flag the entire anomaly window.
                        elif int(anomaly_window_type) == 2:
                            FN_LS += 1 #Failed to flag the entire anomaly window.
                        elif int(anomaly_window_type) == 3:
                            FN_TC += 1 #Failed to flag the entire anomaly window.
                        elif int(anomaly_window_type) == 4:
                            FN_LTO += 1 #Failed to flag the entire anomaly window.
                        elif int(anomaly_window_type) == 5:
                            FN_SALO += 1 #Failed to flag the entire anomaly window.
                        else:
                            print("ERROR ANOMALY WINDOW HAD NO RECOGNIZED TYPE!")
                            print(anomaly_window_type)
                    else:
                        has_flagged_anomaly_window = False

                    reaction_counter = 0.
                    anomaly_window_type = 0
                #Entering an anomaly window.
                if D_truth[i][t] != 0 and not in_anomaly_window:
                    in_anomaly_window = True
                    anomaly_window_type = D_truth[i][t]


                if guessed_anomalies[i][t] == 0 and not in_anomaly_window:
                    TN += 1 #Correct to not guess for an anomaly.
                if guessed_anomalies[i][t] != 0 and in_anomaly_window and not has_flagged_anomaly_window:
                    TP += 1 #Correct guess, within anomaly window.
                    tot_reaction += reaction_counter #Add reaction counter to av
                    reaction_counter = 0.
                    has_flagged_anomaly_window = True
                    if int(anomaly_window_type) == 1:
                        TP_ALO += 1 
                    elif int(anomaly_window_type) == 2:
                        TP_LS += 1 
                    elif int(anomaly_window_type) == 3:
                        TP_TC += 1 
                    elif int(anomaly_window_type) == 4:
                        TP_LTO += 1 
                    elif int(anomaly_window_type) == 5:
                        TP_SALO += 1 
                    else:
                        print("ERROR ANOMALY WINDOW  HAD NO TYPE!")
                        print(anomaly_window_type)
                if guessed_anomalies[i][t] != 0 and not in_anomaly_window:
                    FP += 1 #Erroneous guess, outside anomaly window.
                if guessed_anomalies[i][t] == 0 and in_anomaly_window and not has_flagged_anomaly_window:
                    reaction_counter += 1 #Failed to react to an anomaly which was present in the current timestep.

        print(" ")
        print("|" + "TP: " + str(TP) + "|TN: " + str(TN) +  "|" + "FP: " + str(FP) + "|" + "FN: " + str(FN) + "|")
        if TP == 0: 
            print("Somethings is wrong. No TP's")
            TP = 0.0000001
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

        
        avg_reaction = tot_reaction/(TP)
        


        F1 = 2* (precision * recall) / (precision + recall)
        print("Anomalies:   " + str(TP + FN))
        print("Guesses:     " + str(TP + FP))
        print("precision:   " + str(precision))
        print("recall:      " + str(recall))
        print("F1:          " + str(F1))
        print("Reaction:    " + str(avg_reaction))
        print("TP_ALO:      " + str(TP_ALO))
        print("FN_ALO:      " + str(FN_ALO))
        print("TP_LS:       " + str(TP_LS))
        print("FN_LS:       " + str(FN_LS))
        print("TP_TC:       " + str(TP_TC))
        print("FN_TC:       " + str(FN_TC))
        print("TP_LTO:      " + str(TP_LTO))
        print("FN_LTO:      " + str(FN_LTO))
        print("TP_SALO:     " + str(TP_SALO))
        print("FN_SALO:     " + str(FN_SALO))




        with open(str(dataType) + "_" + str(dataSize) + '_result_lstm.txt', 'a') as the_result_file:
            the_result_file.write("\n\n\n")
            the_result_file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
            the_result_file.write("\n")
            the_result_file.write("|" + "TP: " + str(TP) + "|TN: " + str(TN) +  "|" + "FP: " + str(FP) + "|" + "FN: " + str(FN) + "|")
            the_result_file.write("\nAnomalies:   " + str(TP + FN))
            the_result_file.write("\nGuesses:     " + str(TP + FP))
            the_result_file.write("\nprecision:   " + str(precision))
            the_result_file.write("\nrecall:      " + str(recall))
            the_result_file.write("\nF1:          " + str(F1))
            the_result_file.write("\nReaction:    " + str(avg_reaction))
            the_result_file.write("\nTP_ALO:      " + str(TP_ALO))
            the_result_file.write("\nFN_ALO:      " + str(FN_ALO))
            the_result_file.write("\nTP_LS:       " + str(TP_LS))
            the_result_file.write("\nFN_LS:       " + str(FN_LS))
            the_result_file.write("\nTP_TC:       " + str(TP_TC))
            the_result_file.write("\nFN_TC:       " + str(FN_TC))
            the_result_file.write("\nTP_LTO:      " + str(TP_LTO))
            the_result_file.write("\nFN_LTO:      " + str(FN_LTO))
            the_result_file.write("\nTP_SALO:     " + str(TP_SALO))
            the_result_file.write("\nFN_SALO:     " + str(FN_SALO))

        return guessed_anomalies, all_preds, all_likelihoods, all_residuals

        



    def plotAnomalyDetection(self, D, D_truth, D_unpolluted, D_guesses, D_preds, D_likelihoods, D_residuals):
        if D_guesses is None:
            D_guesses = np.zeros((len(D), len(D[0])))

        
        for sample_nr in range(min(len(D),10)):

            x = range(len(D[sample_nr]))

            at = np.trim_zeros(D_truth[sample_nr])
            
            

            plt.figure(1)
            plt.subplot(311)
            line_mean, = plt.plot(D_preds[sample_nr], 'g-', label='LSTM prediction')
            all_gi = np.argwhere(D_guesses[sample_nr] != 0) #all guesses
            all_ai = np.argwhere(D_truth[sample_nr] != 0)   #all actuall anomalies
            for gi in all_gi:
                dot_wrong, = plt.plot(gi, D[sample_nr][gi], 'yo', label='Incorrect Guess')
                #plt.axvspan(ai-0.1, ai+0.1, facecolor='#990000', alpha=0.8)
            for ai in all_ai:
                dot_actual, = plt.plot(ai, D[sample_nr][ai], 'ro', label='Anomaly')
                if ai in all_gi:
                    dot_correct, = plt.plot(ai, D[sample_nr][ai], 'ko', label='Correct Guess')
                #plt.axvspan(ai-0.1, ai+0.1, facecolor='#990000', alpha=0.8)
            #plt.fill_between(x, (D_preds[sample_nr] - 1), (D_preds[sample_nr] + 1), facecolor='green', alpha=0.4)
            line_actual, = plt.plot(D[sample_nr], 'r-', label='True Values')
            line_unpolluted, = plt.plot(D_unpolluted[sample_nr], label='True Values without Anomalies')
            plt.axvline(x=int(len(D[sample_nr])*0.1), color='grey')

            plt.title("LSTM on anomlies that are present in this serie: " + str(np.unique(at)))
            plt.legend(handles=[line_mean, line_actual, line_unpolluted, dot_wrong, dot_actual, dot_correct])
            

            plt.subplot(312)

            line_al, = plt.plot(D_likelihoods[sample_nr], 'k-', label='Anomaly Likelihood')
            plt.ylim((0.995, 1))
            plt.legend(handles=[line_al])
            
            plt.subplot(313)
            plt.title("Residuals")
            line_resi, = plt.plot(D_residuals[sample_nr], 'y-', label='Residuals')
            plt.legend(handles=[line_resi])
            
            plt.show()


