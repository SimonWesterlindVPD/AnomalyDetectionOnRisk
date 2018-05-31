import numpy as np
import matplotlib.pyplot as plt
from random import randint
import math

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

       
    return ys_price[200:]

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

    return ys_price[200:]


#Calculates the Historical Value at Risk of the series over the period provided
#Expected the serie to be daily returns.
def return_to_VaR95(returns, period):
    VaR_serie = []
    for day in range(period, len(returns)):
        VaR = returns[day-period:day].std() * np.sqrt(period) * 1.645
        VaR_serie.append(VaR)

    return VaR_serie[200:]


def add_anomaly_ALO(ts):
    #print("ALO")
    ts_to_pollut = np.copy(ts)
    anomaly_index = randint(0,len(ts))
    anomaly_length = randint(1,100)
    
    shift = 0.45
    std = np.std(ts)


    if randint(0,1) == 1: 
        for i in range(anomaly_index, min(anomaly_index + anomaly_length, len(ts))):
            ts_to_pollut[i] -= 5*std
    else:
        for i in range(anomaly_index, min(anomaly_index + anomaly_length, len(ts))):
            ts_to_pollut[i] += 5*std

    anomaly_indeces = range(anomaly_index, min(anomaly_index + anomaly_length, len(ts)))
    return ts_to_pollut, 1, anomaly_indeces

def add_anomaly_LS(ts):
    #print("LS")
    ts_to_pollut = np.copy(ts)
    anomaly_index = randint(0,len(ts))
    anomaly_length = 1

    shift = 0.45
    std = np.std(ts)
    
    if randint(0,1) == 1: 
        for i in range(anomaly_index, len(ts)):
            ts_to_pollut[i] -= 5*std
    else:
        for i in range(anomaly_index, len(ts)):
            ts_to_pollut[i] += 5*std

    anomaly_indeces = range(anomaly_index, min(anomaly_index + anomaly_length, len(ts)))
    return ts_to_pollut, 2, anomaly_indeces



def add_anomaly_TC(ts):
    #print("TC")
    ts_to_pollut = np.copy(ts)
    anomaly_index = randint(0,len(ts))
    anomaly_length = 30
    δ = 0.97

    shift = 0.45
    std = np.std(ts)

    iter = 0
    if randint(0,1) == 1: 
        for i in range(anomaly_index, len(ts)):
            iter += 1
            ts_to_pollut[i] -= 5*std*(δ**iter)
    else:
        for i in range(anomaly_index, len(ts)):
            iter += 1
            ts_to_pollut[i] += 5*std*(δ**iter)

    anomaly_indeces = range(anomaly_index, min(anomaly_index + anomaly_length, len(ts)))
    return ts_to_pollut, 3, anomaly_indeces

def add_anomaly_LTO(ts):
    #print("LTO")
    ts_to_pollut = np.copy(ts)
    anomaly_index = randint(0, len(ts))
    anomaly_length = 30
    func_var = 0.0

    shift = 0.45
    std = np.std(ts)

    if randint(0,1) == 1: 
        for i in range(anomaly_index, len(ts)):
            func_var += 0.1
            ts_to_pollut[i] -= 5*std*LTO_increase_function(func_var)
    else:
        for i in range(anomaly_index, len(ts)):
            func_var += 0.1
            ts_to_pollut[i] += 5*std*LTO_increase_function(func_var)

    anomaly_indeces = range(anomaly_index, min(anomaly_index + anomaly_length, len(ts)))
    return ts_to_pollut, 4, anomaly_indeces

def add_anomaly_SALO(ts):
    #print("SALO")

    ts_to_pollut = np.copy(ts)
    anomaly_season = 30
    anomaly_index = randint(0,anomaly_season)
    anomaly_length = randint(1,10)
    shift = 0.45
    std = np.std(ts)

    anomaly_indeces = []
    if randint(0,1) == 1: 
        for i in range(len(ts)):
            if  i%anomaly_season in range(0, anomaly_length):
                anomaly_indeces.append(i)
                ts_to_pollut[i] -= 5*std
    else:
        for i in range(len(ts)):
            if  i%anomaly_season in range(0, anomaly_length):
                anomaly_indeces.append(i)
                ts_to_pollut[i] += 5*std
    return ts_to_pollut, 5, anomaly_indeces

#D1 is ARMA(0,0)-GARCH(1,1) returns.
def genD1(number_of_series=500, length_of_series=100): 
    for i in range(number_of_series):
        #Generating. ω = 1 - α - β, makes for an unconditional-GARCH.
        ω = 0.2
        α = [0.2]
        β = [0.6]
        ϵ = garch(ω, α, β, length_of_series)
        time_serie = return_to_price_AR_1(ϵ)

        dice_roll = randint(1,6)
        if dice_roll == 1:
            time_serie_polluted, at, ai = add_anomaly_ALO(time_serie)
        elif dice_roll == 2:
            time_serie_polluted, at, ai = add_anomaly_LS(time_serie)
        elif dice_roll == 3:
            time_serie_polluted, at, ai = add_anomaly_TC(time_serie)
        elif dice_roll == 4:
            time_serie_polluted, at, ai = add_anomaly_LTO(time_serie)
        elif dice_roll == 5:
            time_serie_polluted, at, ai = add_anomaly_SALO(time_serie)
        else:
            #print("Nothing")
            time_serie_polluted, at, ai = time_serie, 0, [0]
        #Plotting
        x = range(len(time_serie))
        #plt.plot(x, time_serie_polluted, 'r')
        #plt.plot(x, time_serie)
        #plt.show()

        #Saving
        if i == 0:
            D = np.array(time_serie_polluted)
            D_unpolluted = np.array(time_serie)
            D_truth = np.zeros(len(time_serie))
            for anom_i in ai:
                D_truth[anom_i] = at
        else:  
            D = np.vstack([D, time_serie_polluted])
            D_unpolluted = np.vstack([D_unpolluted, time_serie])
            D_truth_new = np.zeros(len(time_serie))
            for anom_i in ai:
                D_truth_new[anom_i] = at
            D_truth = np.vstack([D_truth, D_truth_new])
    np.savetxt("D1_" + str(length_of_series) + ".csv", D, delimiter=",")
    np.savetxt("D1_unpolluted_" + str(length_of_series) + ".csv", D_unpolluted, delimiter=",")
    np.savetxt("D1_truth_" + str(length_of_series) + ".csv", D_truth, delimiter=",")

#D2 HVaR20 from ARMA(0,0)-GARCH(1,1) returns. 
def genD2(number_of_series=500, length_of_series=100): 
    for i in range(number_of_series):
        #Generating
        ω = 0.5
        α = [0.2]
        β = [0.6]

        n = 3000
        time_serie_return = garch(ω, α, β, length_of_series, lambda n: np.random.standard_t(123, n))
        time_serie_VaR = return_to_VaR95(time_serie_return, 20)

        dice_roll = randint(1,6)
        #print("dice_roll:" + str(dice_roll))
        if dice_roll == 2:
            time_serie_polluted, at, ai = add_anomaly_LS(time_serie_VaR)
        elif dice_roll == 3:
            time_serie_polluted, at, ai = add_anomaly_TC(time_serie_VaR)
        elif dice_roll == 4:
            time_serie_polluted, at, ai = add_anomaly_LTO(time_serie_VaR)
        else:
           # print("Nothing")
            time_serie_polluted, at, ai = time_serie_VaR, 0, [0]
        #Plotting
        x_VaR = range(len(time_serie_VaR))
        x_returns = range(len(time_serie_return))

        #plt.subplot(211)
        #plt.plot(x_VaR, time_serie_polluted, 'r')
        #plt.plot(x_VaR, time_serie_VaR)
        #plt.subplot(212)
        #plt.plot(x_returns, time_serie_return)
        #plt.show()

        #Saving
        if i == 0:
            D = np.array(time_serie_polluted)
            D_unpolluted = np.array(time_serie_VaR)
            D_truth = np.zeros(len(time_serie_VaR))
            for anom_i in ai:
                D_truth[anom_i] = at
        else:  
            D = np.vstack([D, time_serie_polluted])
            D_unpolluted = np.vstack([D_unpolluted, time_serie_VaR])
            D_truth_new = np.zeros(len(time_serie_VaR))
            #print(ai)
            #print("Anomaly type:" + str(at))

            #print("----------------------")
            for anom_i in ai:

                D_truth_new[anom_i] = at
            D_truth = np.vstack([D_truth, D_truth_new])
    np.savetxt("D2_" + str(length_of_series) + ".csv", D, delimiter=",")
    np.savetxt("D2_unpolluted_" + str(length_of_series) + ".csv", D_unpolluted, delimiter=",")
    np.savetxt("D2_truth_" + str(length_of_series) + ".csv", D_truth, delimiter=",")



#D4 is ARMA(2,2)-GARCH(1,1) returns.
def genD4(number_of_series=500, length_of_series=100): 

    for i in range(number_of_series):
        #Generating 
        ω = 0.2
        α = [0.2]
        β = [0.6]
        ϵ = garch(ω, α, β, length_of_series)
        time_serie_returns = return_to_price_ARMA_2_2(ϵ)
        
        anomaly_or_not = randint(0,1)
        if anomaly_or_not == 1:
            dice_roll = randint(1,4)
            if dice_roll == 1:
                #Time serie has an ALO in the end
                time_serie_polluted, at, ai = add_anomaly_ALO(time_serie_returns)
            elif dice_roll == 2:
                #Time serie has an LS in the end
                time_serie_polluted, at, ai = add_anomaly_LS(time_serie_returns)
            elif dice_roll == 3:
                #Time serie has a TC in the end
                time_serie_polluted, at, ai = add_anomaly_TC(time_serie_returns)
            elif dice_roll == 4:
                #Time serie has an LTO in the end
                time_serie_polluted, at, ai = add_anomaly_LTO(time_serie_returns)
        else:
            #print("Nothing")
            time_serie_polluted, at, ai = time_serie_returns, 0, [0]
        
        #Plotting
        """
        x = range(len(time_serie_returns))
        plt.plot(x, time_serie_polluted, 'r')
        plt.plot(x, time_serie_returns)
        plt.show()
        """

        #Saving
        if i == 0:
            D = np.array(time_serie_polluted)
            D_unpolluted = np.array(time_serie_returns)
            D_truth = np.zeros(len(time_serie_returns))
            for anom_i in ai:
                D_truth[anom_i] = at
        else:  
            D = np.vstack([D, time_serie_polluted])
            D_unpolluted = np.vstack([D_unpolluted, time_serie_returns])
            D_truth_new = np.zeros(len(time_serie_returns))
            for anom_i in ai:
                D_truth_new[anom_i] = at
            D_truth = np.vstack([D_truth, D_truth_new])
    np.savetxt("D4_" + str(length_of_series) + ".csv", D, delimiter=",")
    np.savetxt("D4_unpolluted_" + str(length_of_series) + ".csv", D_unpolluted, delimiter=",")
    np.savetxt("D4_truth_" + str(length_of_series) + ".csv", D_truth, delimiter=",")


if __name__ == "__main__":




    print("Generating D1 of serie lenth 100")
    genD1(500, 100)
    print("Generating D1 of serie lenth 500")
    genD1(500, 500)
    print("Generating D1 of serie lenth 1500")
    genD1(500, 1500)
    print("Generating D1 of serie lenth 3000")
    genD1(500, 3000)

    print("Generating D2 of serie lenth 100")
    genD2(500, 100)
    print("Generating D2 of serie lenth 500")
    genD2(500, 500)
    print("Generating D2 of serie lenth 1500")
    genD2(500, 1500)
    print("Generating D2 of serie lenth 3000")
    genD2(500, 3000)

    print("Generating D4 of serie lenth 100")
    genD4(500, 100)
    print("Generating D4 of serie lenth 500")
    genD4(500, 500)
    print("Generating D4 of serie lenth 1500")
    genD4(500, 1500)
    print("Generating D4 of serie lenth 3000")
    genD4(500, 3000)




    print("Finished!")

