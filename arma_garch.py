import csv
import os
import sys
import datetime
import re

import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import sklearn as sk

import datetime

import subprocess

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from arch import arch_model

from numpy import genfromtxt
from operator import add, sub

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import mean_squared_error

p = print


#The EWMA class is capable of fetching data, performing anomaly detection on said data, calculates the score and plot the result.
class arma_garch():

    #Models a ARIMA(p,q,d)-Garch(i,j) after the series in D and then forcasts the last 5 points of each.
    def ARIMA_GARCH(self, data, dataType, dataSize):
        X = data


        #Performs an Augmented Dickey-Fuller test to check for stationarity.
        result = smt.adfuller(X[0])
        pvalue = result[1]
        if pvalue < 0.2:
            print('p-value = ' + str(pvalue) + ' The series is likely stationary.')
            differentiation = "None" 
        else:
            print('p-value = ' + str(pvalue) + ' The series is likely NON-stationary.')
            #differentiation = "Once" 
            differentiation = "None"
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))



        #Starts a subprocess of the ./arima_garch.r program which is a program written in R and uses the rugarch packaged to perform anomaly detection using an ARMA-GARCH
        nr_of_series = len(data)
        print("Starting subprocess")
        subprocess.call(["Rscript", "--vanilla", "./arima_garch.r", str(dataType), str(dataSize), str(nr_of_series), differentiation])
        print("Ended subprocess")


        #Load the results that were created by the R program, from file. 
        forecasts_mean = genfromtxt('forecasts_mean_' + dataType + "_" + dataSize + '.csv', skip_header=0, delimiter=' ', dtype=float)
        forecasts_variance = genfromtxt('forecasts_variance_' + dataType + "_" + dataSize + '.csv', skip_header=0, delimiter=' ', dtype=float)


        




        #Correcting for NaN's and 0's
        for row in range(len(forecasts_mean)):
            for col in range(len(forecasts_mean[row])):
                if forecasts_mean[row][col] == 0 and col > 0:
                    forecasts_mean[row][col] = forecasts_mean[row][col-1]
                if np.isnan(forecasts_variance[row][col]):
                    if col > 0:
                        forecasts_variance[row][col] = forecasts_variance[row][col-1]
                    else:
                        forecasts_variance[row][col] = 1

        
        return forecasts_mean, forecasts_variance





    #Expected a tensorflow matrix of datapoints.
    def runAnomalyDetection(self, D, D_truth, D_unpolluted, dataType, dataSize):

        data = D
        α = 4

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

        startup_period = int(0.1*len(D[0]))

        print("Running ARIMA-GARCH-AD")
        guessed_anomalies = np.zeros((len(D), len(D[0])))
        forecasts_mean, forecasts_variance = self.ARIMA_GARCH(D, dataType, dataSize)

        all_preds = np.zeros((len(D), len(D[0])))
        all_residuals = np.zeros((len(D), len(D[0])))
        for i in range(len(D)):
            #Make guesses
            for t in range(startup_period, len(D[i])):

                UCL = forecasts_mean[i][t-startup_period] + α * forecasts_variance[i][t-startup_period]
                LCL = forecasts_mean[i][t-startup_period] - α * forecasts_variance[i][t-startup_period]
                if  UCL < D[i][t] or D[i][t] < LCL:
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
                            print("ERROR ANOMALY WINDOW  HAD NO TYPE!")
                            print(int(anomaly_window_type))
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
                        print(int(anomaly_window_type))
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


        with open(str(dataType) + "_" + str(dataSize) + '_result_arma.txt', 'a') as the_result_file:
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

        return guessed_anomalies, forecasts_mean, forecasts_variance

    def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
        print("In tsplot")
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        with plt.style.context(style):
            fig = plt.figure(figsize=figsize)
            #mpl.rcParams['font.family'] = 'Ubuntu Mono'
            layout = (3, 2)
            ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
            acf_ax = plt.subplot2grid(layout, (1, 0))
            pacf_ax = plt.subplot2grid(layout, (1, 1))
            qq_ax = plt.subplot2grid(layout, (2, 0))
            pp_ax = plt.subplot2grid(layout, (2, 1))

            y.plot(ax=ts_ax)
            ts_ax.set_title('Time Series Analysis Plots')
            smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
            smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
            sm.qqplot(y, line='s', ax=qq_ax)
            qq_ax.set_title('QQ Plot')
            scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

            plt.tight_layout()
            plt.show()
        return



    #Finds parameters p, d, q for ARIMA(p, d, q) which maximizes MLE-BIC for the data serie X.
    #The model cannot try more parameters than #obs/#params = 5
    def find_best_order(X):
        best_bic = np.inf
        best_order = None
        best_mdl = None


        max_p = 1
        max_d = 2
        max_q = 1


        #Creates the biggest range possible for the gridsearch with likely convergance.
        while len(X) > 5*(max_p + max_d + max_q) or max_p > 10:
            max_p += 1
            max_q += 1


        p_rng = range(max_p) # [0, 1, 2,..., max_p]
        q_rng = range(max_d) # [0, 1]
        d_rng = range(max_q) # [0, 1, 2,..., max_q]
        for i in p_rng:
            for d in d_rng:
                for j in q_rng:
                    try:
                        tmp_mdl = smt.ARIMA(X, order=(i,d,j)).fit(method='mle', disp=0, trend='nc')
                        tmp_bic = tmp_mdl.bic
                        if tmp_bic < best_bic:
                            best_bic = tmp_bic
                            best_order = (i, d, j)
                            best_mdl = tmp_mdl
                    except: continue


        p('bic: {:6.5f} | order: {}'.format(best_bic, best_order))
        # aic: -11518.22902 | order: (4, 0, 4)

        # ARIMA model resid plot
        #_ = tsplot(best_mdl.resid, lags=30)
        print("best_order: " + str(best_order))
        return best_order






    def plotAnomalyDetection(self, D, D_truth, D_unpolluted, D_guesses, D_means, D_vars):
        for row in range(min(len(D), 10)):


            plt.figure(1)
            plt.subplot(211)
            x = range(len(D_means[0]))
            line_mean, = plt.plot(D_means[row], 'g', label='Conditional Mean')
            plt.fill_between(x, (D_means[row] - 3*D_vars[row]), (D_means[row] + 3*D_vars[row]), facecolor='green', alpha=0.4)
            line_actual, = plt.plot(D[row, int(0.1*len(D[row])):], 'r', label='True Values')
            line_unpolluted, = plt.plot(D_unpolluted[row, int(0.1*len(D[row])):], label='True Values without Anomalies')
            all_gi = np.argwhere(D_guesses[row] != 0) #all guesses
            all_ai = np.argwhere(D_truth[row] != 0)   #all actuall anomalies
            for gi in all_gi:
                dot_wrong, = plt.plot(gi-int(0.1*len(D[row])), D[row][gi], 'yo', label='Incorrect Guess')
                #plt.axvspan(ai-0.1, ai+0.1, facecolor='#990000', alpha=0.8)
            for ai in all_ai:
                dot_actual, = plt.plot(ai-int(0.1*len(D[row])), D[row][ai], 'ro', label='Anomaly')
                if ai in all_gi:
                    dot_correct, = plt.plot(ai-int(0.1*len(D[row])), D[row][ai], 'ko', label='Correct Guess')
            plt.legend(handles=[line_mean, line_actual, line_unpolluted, dot_wrong, dot_actual, dot_correct])
            


            plt.subplot(212)
            line_variance, = plt.plot(D_vars[row], 'orange', label='Conditional Variance')
            plt.legend(handles=[line_variance])
            
            plt.show()


