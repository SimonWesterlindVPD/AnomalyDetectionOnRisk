# AnomalyDetectionOnRisk
A comparison between 5 models for performing anomaly detection on financial risk measures and returns. These experiments were conducted as part of the degree project Anomaly Detection for Portfolio Risk Management, which can be found in Simon_Westerlind_Masters_Thesis.pdf or on [Diva](www.google.com).
## Prerequisites
1. Install [Conda](https://conda.io/docs/user-guide/install/index.html)
2. Install the conda requirements with
```
conda install --yes --file requirements.txt
```
3. Install the [rugarch](https://cran.r-project.org/web/packages/rugarch/index.html) package. Elsewise the ARMA-GARCH will not work.

4. Install [NuPIC](http://nupic.docs.numenta.org/1.0.3/index.html)

5. Copy the returns_and_risk folder which exists in ./htm and place it in /nupic/examples/opf/clients/

## Run
To run the EWMA, ARMA-GARCH, LSTM and HardLimits, run

```
python garch_long.py
```
while in the ./garch folder. Thereafter run
```
python run.py --plot
```

The HTM can be run with the same command from within /nupic/examples/opf/clients/returns_and_risk/anomaly/one_returns. However first you must create a separate conda environment and install:
´´´
conda install --yes --file requirements_htm.txt
´´´
