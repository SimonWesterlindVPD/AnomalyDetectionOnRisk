import numpy as np
import matplotlib.pyplot as plt
from random import randint, uniform
import math
from numpy import genfromtxt


'''
This program creates time series. The time series can be created with varying length.
The time series are inserted with four types of anomalies. Additive Level Outliers, Level Shift Outliers, Trancient Change Outliers and Local Trend Outliers.

'''

def getData(datatype="D1", size="100"):
    D = genfromtxt('./' + datatype + '_' + size + '.csv', delimiter=',')
    D_no_anomalies = genfromtxt('./' + datatype + '_unpolluted_' + size + '.csv', delimiter=',')
    D_truth = genfromtxt('./' + datatype + '_truth_' + size + '.csv', delimiter=',')
    return D, D_no_anomalies, D_truth


def checkD1():
    D, D_no_anomalies, D_truth = getData("D1", "5000")
    print("D1 has nan: " + str(np.isnan(D).any()))

def checkD2():
    D, D_no_anomalies, D_truth = getData("D2", "5000")
    print("D2 has nan: " + str(np.isnan(D).any()))

def checkD4():
    D, D_no_anomalies, D_truth = getData("D4", "5000")
    print("D4 has nan: " + str(np.isnan(D).any()))

def fixD1():
    D, D_no_anomalies, D_truth = getData("D1", "5000")
    print("has nan: " + str(np.isnan(D).any()))

    list_of_indecies = [162, 237]
    for ind in list_of_indecies:
        ω = 0.2
        α = [0.2]
        β = [0.6]
        ϵ = garch(ω, α, β, 5000)
        freq = 1/1000


        
        time_serie = return_to_price_AR_1(ϵ)
        truth = np.zeros(len(time_serie))
        time_serie_polluted = time_serie

        for t in range(len(time_serie)):
            anomaly_roll = uniform(0, 1)
            if anomaly_roll < freq:
                
                dice_roll = randint(1,4)
                if dice_roll == 1:
                    #Time serie has an LS in the end
                    time_serie_polluted, at, ai = add_anomaly_LS(time_serie_polluted, time_serie, t, 5)
                elif dice_roll == 2:
                    #Time serie has a TC in the end
                    time_serie_polluted, at, ai = add_anomaly_TC(time_serie_polluted, time_serie, t, 5)
                elif dice_roll == 3:
                    #Time serie has an LTO in the end
                    time_serie_polluted, at, ai = add_anomaly_ALO(time_serie_polluted, time_serie, t, 8)
                elif dice_roll == 4:
                    #Time serie has an LTO in the end
                    time_serie_polluted, at, ai = add_anomaly_LTO(time_serie_polluted, time_serie, t, 8)

                for anom_i in ai:
                    truth[anom_i] = at

                #If we have had an anomaly we do not want one directly again after.
                t += 50
        #Plotting
        
        x = range(len(time_serie))


        plt.plot(x, time_serie)
        plt.xlabel('Days')
        plt.ylabel('Price')    
        plt.show()

        plt.plot(x, time_serie_polluted, 'r')
        plt.plot(x, time_serie)
        for truth_index in range(len(truth)):
            if truth[truth_index] > 0:
                plt.plot(truth_index, time_serie_polluted[truth_index], 'rx')
        


        plt.xlabel('Days')
        plt.ylabel('Price')     
        plt.show()
        

        #Saving
        D[ind] = time_serie_polluted
        D_no_anomalies[ind] = time_serie
        D_truth[ind] = truth


    print("has nan: " + str(np.isnan(D).any()))


    np.savetxt("D1_" + str(5000) + ".csv", D, delimiter=",")
    np.savetxt("D1_unpolluted_" + str(5000) + ".csv", D_no_anomalies, delimiter=",")
    np.savetxt("D1_truth_" + str(5000) + ".csv", D_truth, delimiter=",")




def garch(ω, α, β, n_out=3000, err_distr=None):
    """
    Returns an array of `n_out` number of data points generated from a
    `GARCH(p, q)` process determined by the given parameters ω, α and β, with
    `p = len(α)` and `q = len(β)`.

    The error terms `ɛ` are taken from the distribution `err_distr` which defaults
    to N(0, 1). The `err_distr` should be given as a function `err_distr(n)`
    where `n` is the number of samples drawn from the distribution, eg
    `err_distr = lambda n: np.random.normal(0, 1, n)`.
    """
    p = len(α)
    q = len(β)

    # Since the first max(p, q) number of points are not generated from the garch
    # process, the first points are garbage (extending beyond max(p, q)),
    # so we drop n_pre > max(p, q) number of points.(Now drops first 200 elements when returns are generated.)
    n_pre = max(p, q) + 200
    n = n_pre + n_out

    # Sample noise
    if not err_distr:
        err_distr = lambda n: np.random.normal(0, 1, n)

    ɛ = err_distr(n)

    y = np.zeros(n)
    σ = np.zeros(n)

    # Pre-populate first max(p, q) values, because they are needed in the iteration.
    for k in range(max(p, q)):
        σ[k] = np.random.normal(0, 1)
        y[k] = σ[k] * ɛ[k]

    # Run the garch process, notation from
    # http://stats.lse.ac.uk/fryzlewicz/lec_notes/garch.pdf
    for k in range(max(p, q), n):
        α_term = sum([α[i] * y[k-i]**2 for i in range(p)])
        β_term = sum([β[i] * σ[k-i]**2 for i in range(q)])
        σ[k] = np.sqrt(ω + α_term + β_term)
        y[k] = σ[k] * ɛ[k]

    return y

def LTO_increase_function(x):
    if x < 0:
        return 0.
    elif x > 1:
        return 1.
    else:
        return x

def LTO_increase_function_sigmoidal(x):
    if x < 0:
        return 0.
    elif x > 1:
        return 1.
    else:
        return x

# Make a cumulative series from a "delta series".
def delta_to_cum(ys):
    ys_cum = []
    y_cum = 0
    for y in ys:
        y_cum += y
        ys_cum.append(y_cum)
    return ys_cum

# This price follows an ARIMA(0,0,0)-GARCH(1,1) model.
def return_to_price_AR_1(returns):
    ys_price = []
    y_price_yesterday = 0 #Starting price get to be 0. Rest is just scaling factor.
    for r in returns:
        y_price = y_price_yesterday + r
        ys_price.append(y_price)
        #print("---------------------------------------")
        #print("y: " + str(r))
        #print("y_price: " + str(y_price))
        #print("y_price_yesterday: " + str(y_price_yesterday))

        y_price_yesterday = y_price

       
    return ys_price[201:]

def return_to_price_ARMA_2_2(ϵ):

    #THIS needs to be done in DELTA TERMS and translated into price

    ys_price = []
    ys_price.append(0.1) #price two days ago.
    ys_price.append(0.2) #price yesterday.

    #there are 200 extra values in ϵ and also in ys_price. They are cut off at the end.
    for t in range(2,len(ϵ)):
        y_price_today = 0.4*ys_price[t-1] + 0.25*ys_price[t-2] + 0.15*ϵ[t-1] + 0.1*ϵ[t-2] + 1*ϵ[t] #y(t) = AR(0.8, 0.2) + MA(0.3, 0.2) + GARCH(1,1)
        ys_price.append(y_price_today)   


    #plt.plot(ys_price)
    #plt.show()

    return ys_price[201:]


#Calculates the Historical Value at Risk of the series over the period provided
#Expected the serie to be daily returns.
def return_to_VaR95(returns, period):
    VaR_serie = []
    for day in range(period, len(returns)):
        VaR = returns[day-period:day].std() * np.sqrt(period) * 1.645
        VaR_serie.append(VaR)

    return VaR_serie[201:]


def add_anomaly_ALO(ts, tr, t, magnitude = 3, anomaly_max_length = 50):

    #print("ALO")
    ts_to_pollut = np.copy(ts)
    anomaly_index = t
    anomaly_length = randint(1,anomaly_max_length)
    
    
    relevance_period = 100

    shift = 0.45
    #std = np.std(tr)
    std = np.std(tr[max(0, anomaly_index-relevance_period):anomaly_index])



    if randint(0,1) == 1: 
        for i in range(anomaly_index, min(anomaly_index + anomaly_length, len(ts))):
            ts_to_pollut[i] -= magnitude*std
    else:
        for i in range(anomaly_index, min(anomaly_index + anomaly_length, len(ts))):
            ts_to_pollut[i] += magnitude*std

    anomaly_indeces = range(anomaly_index, min(anomaly_index + anomaly_length, len(ts)))
    return ts_to_pollut, 1, anomaly_indeces

def add_anomaly_LS(ts, tr, t, magnitude = 5):
    #print("LS")
    ts_to_pollut = np.copy(ts)
    anomaly_index = t
    anomaly_length = 3 #it is still good if we get it soon after.
    relevance_period = 100

    shift = 0.45
    #std = np.std(tr)
    std = np.std(tr[max(0, anomaly_index-relevance_period):anomaly_index])
    
    if randint(0,1) == 1: 
        for i in range(anomaly_index, len(ts)):
            ts_to_pollut[i] -= magnitude*std
    else:
        for i in range(anomaly_index, len(ts)):
            ts_to_pollut[i] += magnitude*std

    anomaly_indeces = range(anomaly_index, min(anomaly_index + anomaly_length, len(ts)))
    return ts_to_pollut, 2, anomaly_indeces



def add_anomaly_TC(ts, tr, t, magnitude = 5, δ = 0.97):
    #print("TC")
    ts_to_pollut = np.copy(ts)
    anomaly_index = t
    anomaly_length = 30
    relevance_period = 100

    shift = 0.45
    #std = np.std(tr)
    std = np.std(tr[max(0, anomaly_index-relevance_period):anomaly_index])

    iter = 0
    if randint(0,1) == 1: 
        for i in range(anomaly_index, len(ts)):
            iter += 1
            ts_to_pollut[i] -= magnitude*std*(δ**iter)
    else:
        for i in range(anomaly_index, len(ts)):
            iter += 1
            ts_to_pollut[i] += magnitude*std*(δ**iter)

    anomaly_indeces = range(anomaly_index, min(anomaly_index + anomaly_length, len(ts)))
    return ts_to_pollut, 3, anomaly_indeces

def add_anomaly_LTO(ts, tr, t, magnitude = 5):
    #print("LTO")
    ts_to_pollut = np.copy(ts)
    anomaly_index = t
    anomaly_length = 30
    func_var = 0.0
    magnitude = 8
    relevance_period = 100

    shift = 0.45
    #std = np.std(tr)
    std = np.std(tr[max(0, anomaly_index-relevance_period):anomaly_index])

    if randint(0,1) == 1: 
        for i in range(anomaly_index, len(ts)):
            func_var += 0.1
            ts_to_pollut[i] -= magnitude*std*LTO_increase_function(func_var)
    else:
        for i in range(anomaly_index, len(ts)):
            func_var += 0.1
            ts_to_pollut[i] += magnitude*std*LTO_increase_function(func_var)

    anomaly_indeces = range(anomaly_index, min(anomaly_index + anomaly_length, len(ts)))
    return ts_to_pollut, 4, anomaly_indeces

def add_anomaly_SALO(ts, tr, t, magnitude = 5):
    #print("SALO")

    ts_to_pollut = np.copy(ts)
    anomaly_season = 30
    anomaly_index = t
    magnitude = 5
    relevance_period = 100
    anomaly_length = randint(1,10)
    shift = 0.45
    #std = np.std(tr)
    std = np.std(tr[max(0, anomaly_index-relevance_period):anomaly_index])

    anomaly_indeces = []
    if randint(0,1) == 1: 
        for i in range(len(ts)):
            if  i%anomaly_season in range(0, anomaly_length):
                anomaly_indeces.append(i)
                ts_to_pollut[i] -= magnitude*std
    else:
        for i in range(len(ts)):
            if  i%anomaly_season in range(0, anomaly_length):
                anomaly_indeces.append(i)
                ts_to_pollut[i] += magnitude*std
    return ts_to_pollut, 5, anomaly_indeces

#D1 is ARMA(0,0)-GARCH(1,1) returns.
def genD1(number_of_series=500, length_of_series=5000): 



    for i in range(number_of_series):
        print("Making serie number: " + str(i))
        #Generating 
        ω = 0.2
        α = [0.2]
        β = [0.6]
        ϵ = garch(ω, α, β, length_of_series)
        freq = 1/1000


        
        time_serie = return_to_price_AR_1(ϵ)
        truth = np.zeros(len(time_serie))
        time_serie_polluted = time_serie

        for t in range(len(time_serie)):
            anomaly_roll = uniform(0, 1)
            if anomaly_roll < freq:
                
                dice_roll = randint(1,4)
                if dice_roll == 1:
                    #Time serie has an LS in the end
                    time_serie_polluted, at, ai = add_anomaly_ALO(time_serie_polluted, time_serie, t, 5)
                elif dice_roll == 2:
                    #Time serie has a TC in the end
                    time_serie_polluted, at, ai = add_anomaly_TC(time_serie_polluted, time_serie, t, 5)
                elif dice_roll == 3:
                    #Time serie has an LTO in the end
                    time_serie_polluted, at, ai = add_anomaly_LS(time_serie_polluted, time_serie, t, 8)
                elif dice_roll == 4:
                    #Time serie has an LTO in the end
                    time_serie_polluted, at, ai = add_anomaly_LTO(time_serie_polluted, time_serie, t, 8)

                for anom_i in ai:
                    truth[anom_i] = at

                #If we have had an anomaly we do not want one directly again after.
                t += 50
        #Plotting
        
        x = range(len(time_serie))


        #plt.plot(x, time_serie)
        #plt.xlabel('Days')
        #plt.ylabel('Price')    
        #plt.show()

        plt.plot(x, time_serie_polluted, )
        #plt.plot(x, time_serie)
        anom_len_counter = 0
        truth_index = 0
        was_anomaly = False
        while truth_index < len(truth):
            if truth[truth_index] > 0:
                while truth[truth_index] > 0 or truth_index == len(time_serie):
                    anom_len_counter += 1
                    truth_index += 1
                x_anom = range(truth_index-anom_len_counter-1, truth_index)
                plt.fill_between(x_anom, -61, 61, facecolor='red', alpha=0.4)
                anom_len_counter=0
                x_anom = 0
            truth_index +=1


        plt.xlabel('Days')
        plt.ylabel('Price')     
        plt.show()
        

        #Saving
        
        if i == 0:
            D = np.array(time_serie_polluted)
            D_unpolluted = np.array(time_serie)
            D_truth = truth
        else:  
            D = np.vstack([D, time_serie_polluted])
            D_unpolluted = np.vstack([D_unpolluted, time_serie])
            D_truth_new = truth
            D_truth = np.vstack([D_truth, D_truth_new])
    np.savetxt("D1_" + str(length_of_series) + ".csv", D, delimiter=",")
    np.savetxt("D1_unpolluted_" + str(length_of_series) + ".csv", D_unpolluted, delimiter=",")
    np.savetxt("D1_truth_" + str(length_of_series) + ".csv", D_truth, delimiter=",")

#D2 HVaR20 from ARMA(0,0)-GARCH(1,1) returns. 
def genD2(number_of_series=500, length_of_series=5000): 

    for i in range(number_of_series):
        print("Series NR:" + str(i))
        #Generating 
        ω = 0.2
        α = [0.2]
        β = [0.6]
        ϵ = garch(ω, α, β, length_of_series)
        freq = 1/1000
        var_history_length = 20


        time_serie_return = garch(ω, α, β, length_of_series + var_history_length, lambda n: np.random.standard_t(123, n))
        time_serie_VaR = return_to_VaR95(time_serie_return, var_history_length)
        truth = np.zeros(len(time_serie_VaR))
        time_serie_polluted = time_serie_VaR

        for t in range(len(time_serie_VaR)):
            anomaly_roll = uniform(0, 1)
            if anomaly_roll < freq:
                
                dice_roll = randint(1,4)
                if dice_roll == 1:
                    #Time serie has an LTO in the end
                    time_serie_polluted, at, ai = add_anomaly_ALO(time_serie_polluted, time_serie_VaR, t, 5)
                elif dice_roll == 2:
                    #Time serie has an LS in the end
                    time_serie_polluted, at, ai = add_anomaly_LS(time_serie_polluted, time_serie_VaR, t, 5)
                elif dice_roll == 3:
                    #Time serie has a TC in the end
                    time_serie_polluted, at, ai = add_anomaly_TC(time_serie_polluted, time_serie_VaR, t, 5)
                elif dice_roll == 4:
                    #Time serie has an LTO in the end
                    time_serie_polluted, at, ai = add_anomaly_LTO(time_serie_polluted, time_serie_VaR, t, 8)

                for anom_i in ai:
                    truth[anom_i] = at

                #If we have had an anomaly we do not want one directly again after.
                t += 50
        #Plotting
        
        x = range(len(time_serie_VaR))

        plt.plot(x, time_serie_polluted)
        #plt.plot(x, time_serie)
        anom_len_counter = 0
        truth_index = 0
        was_anomaly = False
        while truth_index < len(truth):
            if truth[truth_index] > 0:
                while truth[truth_index] > 0 or truth_index == len(time_serie_VaR):
                    anom_len_counter += 1
                    truth_index += 1
                x_anom = range(truth_index-anom_len_counter-1, truth_index)
                plt.fill_between(x_anom, -61, 61, facecolor='red', alpha=0.4)
                anom_len_counter=0
                x_anom = 0
            truth_index +=1


        

        plt.xlabel('Days')
        plt.ylabel('Value-at-Risk') 
        plt.show()
        

        #Saving
        if i == 0:
            D = np.array(time_serie_polluted)
            D_unpolluted = np.array(time_serie_VaR)
            D_truth = truth
        else:  
            D = np.vstack([D, time_serie_polluted])
            D_unpolluted = np.vstack([D_unpolluted, time_serie_VaR])
            D_truth_new = truth
            D_truth = np.vstack([D_truth, D_truth_new])
    np.savetxt("D2_" + str(length_of_series) + ".csv", D, delimiter=",")
    np.savetxt("D2_unpolluted_" + str(length_of_series) + ".csv", D_unpolluted, delimiter=",")
    np.savetxt("D2_truth_" + str(length_of_series) + ".csv", D_truth, delimiter=",")



#D4 is ARMA(2,2)-GARCH(1,1) returns.
def genD4(number_of_series=500, length_of_series=5000): 

    for i in range(number_of_series):
        #Generating 
        ω = 0.2
        α = [0.2]
        β = [0.6]
        ϵ = garch(ω, α, β, length_of_series)
        freq = 1/1000

        time_serie_returns = return_to_price_ARMA_2_2(ϵ)
        truth = np.zeros(len(time_serie_returns))
        time_serie_polluted = time_serie_returns
        for t in range(len(time_serie_returns)):
            anomaly_roll = uniform(0, 1)
            if anomaly_roll < freq:
                
                dice_roll = randint(1,4)
                if dice_roll == 1:
                    #Time serie has an ALO in the end
                    time_serie_polluted, at, ai = add_anomaly_ALO(time_serie_polluted, time_serie_returns, t, 5)
                elif dice_roll == 2:
                    #Time serie has an LS in the end
                    time_serie_polluted, at, ai = add_anomaly_LS(time_serie_polluted, time_serie_returns, t, 5)
                elif dice_roll == 3:
                    #Time serie has a TC in the end
                    time_serie_polluted, at, ai = add_anomaly_TC(time_serie_polluted, time_serie_returns, t, 5)
                elif dice_roll == 4:
                    #Time serie has an LTO in the end
                    time_serie_polluted, at, ai = add_anomaly_LTO(time_serie_polluted, time_serie_returns, t, 8)

                for anom_i in ai:
                    truth[anom_i] = at

                #If we have had an anomaly we do not want one directly again after.
                t += 50
        #Plotting
        
        x = range(len(time_serie_returns))

        x = range(len(time_serie_returns))

        plt.plot(x, time_serie_polluted)
        #plt.plot(x, time_serie)
        anom_len_counter = 0
        truth_index = 0
        was_anomaly = False
        while truth_index < len(truth):
            if truth[truth_index] > 0:
                while truth[truth_index] > 0 or truth_index == len(time_serie_returns):
                    anom_len_counter += 1
                    truth_index += 1
                x_anom = range(truth_index-anom_len_counter-1, truth_index)
                plt.fill_between(x_anom, -61, 61, facecolor='red', alpha=0.4)
                anom_len_counter=0
                x_anom = 0
            truth_index +=1

   
        

        plt.xlabel('Days')
        plt.ylabel('Price') 
        plt.show()
        

        #Saving
        if i == 0:
            D = np.array(time_serie_polluted)
            D_unpolluted = np.array(time_serie_returns)
            D_truth = truth
        else:  
            D = np.vstack([D, time_serie_polluted])
            D_unpolluted = np.vstack([D_unpolluted, time_serie_returns])
            D_truth_new = truth
            D_truth = np.vstack([D_truth, D_truth_new])
    np.savetxt("D4_" + str(length_of_series) + ".csv", D, delimiter=",")
    np.savetxt("D4_unpolluted_" + str(length_of_series) + ".csv", D_unpolluted, delimiter=",")
    np.savetxt("D4_truth_" + str(length_of_series) + ".csv", D_truth, delimiter=",")


def demonstartiveplots():
    #Generating 
    ω = 0.1
    α = [0.2]
    β = [0.6]
    ϵ = garch(ω, α, β, 1825)


    
    time_serie = return_to_price_AR_1(ϵ)
    truth = np.zeros(len(time_serie))
    time_serie_polluted = time_serie

    time_serie_polluted, at, ai = add_anomaly_LS(time_serie_polluted, time_serie, 1800, 8)
    #Plotting
    
    x = range(len(time_serie))


    plt.plot(x, time_serie_polluted)
    #plt.plot(x, time_serie)
    for truth_index in range(len(truth)):
        if truth[truth_index] > 0:
            #plt.plot(truth_index, time_serie_polluted[truth_index], 'rx')
            pass
    


    plt.xlabel('Days')
    plt.ylabel('Value-at-Risk')     
    plt.show()
    

    plt.plot(x, time_serie_polluted, 'r')
    plt.plot(x, time_serie)
    plt.xlabel('Days')
    plt.ylabel('Value-at-Risk')     
    plt.show()    





if __name__ == "__main__":




    
    print("Generating D1 of serie length 5000")
    genD1(500, 5000)
    print("Generating D2 of serie length 5000")
    genD2(500, 5000)
    print("Generating D4 of serie length 5000")
    genD4(500, 5000)
    


    
    print("Finished!")

