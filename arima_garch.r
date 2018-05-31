#"quantmod---------------------------------------------------------------------"
#install.packages("quantmod", repos='https://ftp.acc.umu.se/mirror/CRAN/')
#"lattice---------------------------------------------------------------------"
#install.packages("lattice", repos='https://ftp.acc.umu.se/mirror/CRAN/')
#"timeSeries---------------------------------------------------------------------"
#install.packages("timeSeries", repos='https://ftp.acc.umu.se/mirror/CRAN/')
#"rugarch---------------------------------------------------------------------"
#install.packages("rugarch", repos='https://ftp.acc.umu.se/mirror/CRAN/')
"Packages installed."



# defining a function
is.installed <- function(mypkg) is.element(mypkg, installed.packages()[,1]) 

checkNecessaryPackages <- function(){
	"timeSeries:"
	library("timeSeries")
	is.installed('timeSeries') 
	"quantmod:"
	library("quantmod")
	is.installed('quantmod') 
	"lattice:"
	library("lattice")
	is.installed('lattice') 
	"nloptr:"
	library("nloptr")
	is.installed('nloptr') 
	"rugarch:"
	library("rugarch")
	is.installed('rugarch') 

}

library("rugarch")
#print("RUGARCH IS INSTALLED:")
is.installed('rugarch') 


args <- commandArgs(trailingOnly = TRUE)
datatype <- args[1]
size <- args[2]
nr_of_series <- args[3]
differentiation <- args[4]

#print(datatype)
#print(size)

file_name = paste("garch/", datatype, "_", size, ".csv", sep="")
#print(file_name)

#Convert strings to intergers
size_string = size
size = strtoi(size)
nr_of_series = strtoi(nr_of_series)

return_series = read.csv(file_name, header=FALSE)

tmp_serie = return_series[1,]
tmp_time_serie = as.ts(tmp_serie)
windowLength = as.integer(length(tmp_time_serie)*0.10)

forecasts_for_all_series=matrix(nrow=nr_of_series, ncol=size-windowLength, byrow = TRUE)        # fill matrix by rows 
sigmas_for_all_series =matrix(nrow=nr_of_series, ncol=size-windowLength, byrow = TRUE)        # fill matrix by rows 
##print(forecasts_for_all_series)




#Judge Order


#final.bic <- Inf
final.order <- c(2,0,2)

#first_serie = return_series[1,]
#first_serie_window = as.ts(first_serie[(1):(windowLength)])
#for (p in 0:5) for (q in 0:5) {
#    if ( p == 0 && q == 0) {
#       next
#    }
#
#    arimaFit = tryCatch( arima(first_serie_window, order=c(p, 0, q)),
#                         error=function( err ) FALSE,
#                         warning=function( err ) FALSE )
#    ##print(arimaFit)
#    if( !is.logical( arimaFit ) ) {
#    	##print('actually managed to fit')
#    	##print(p)
#    	##print(q)
#        current.bic <- BIC(arimaFit)
#        if (current.bic < final.bic) {
#            final.bic <- current.bic
#            final.order <- c(p, 0, q)
#            final.arima <- arima(first_serie_window, order=final.order)
#        }
#    } else {
#    	##print('didnt managed to fit')
#        next
#    }
#}

print("Final order: ")
print(final.order)

#Start modellling

for (serie_nr in 1:nr_of_series) {
	one_serie = return_series[serie_nr,]
	one_time_serie = as.ts(one_serie)
	if(differentiation == "Once") {

		one_time_serie = c(0, diff(one_time_serie[1:length(one_time_serie)]))
	}

	foreLength = length(one_time_serie) - windowLength - 1
	forecasts <- vector(mode="character", length=foreLength)
	sigmas <- vector(mode="character", length=foreLength)
	truths <- vector(mode="character", length=foreLength)
	spec = ugarchspec(
        variance.model=list(garchOrder=c(1,1)),
        mean.model=list(armaOrder=final.order, include.mean=TRUE),
        #mean.model=list(armaOrder=c(final.order[1], final.order[3]), include.mean=TRUE),
        distribution.model="sged")

	d = 0
	max_roll = 100
	while (d < foreLength) {
		one_serie_window = as.ts(one_time_serie[(1+d):(windowLength+d)])
		datapoints_left_in_serie = foreLength - d
		nr_of_rolls = min(datapoints_left_in_serie, max_roll)
		one_serie_window_future = as.ts(one_time_serie[(1+d):(windowLength+d+nr_of_rolls)])
		#one_diffed_serie_window = one_serie_window #diff(one_serie_window)


		
	    #spec = ugarchspec(
	    #            variance.model=list(garchOrder=c(1,1)),
	    #            mean.model=list(armaOrder=c(1, 1), include.mean=TRUE),
	    #            #mean.model=list(armaOrder=c(final.order[1], final.order[3]), include.mean=TRUE),
	    #            distribution.model="sged")

			

		model<-ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1), variance.targeting=TRUE), 
		mean.model = list(armaOrder = c(1, 1), include.mean = TRUE), 
		distribution.model = "norm")

		modelfit<-ugarchfit(spec=model,data=one_serie_window, solver='hybrid')
		#print(coef(modelfit))
		if(is.null(coef(modelfit))){
			print("FAILED to CONVERGE")
			#Handles exception naively
			forecasts[d+1] = 0
			sigmas[d+1] = 1

			d = d + 1

		}
		else{
			spec = getspec(modelfit);
			setfixed(spec) <- as.list(coef(modelfit))
			fore = ugarchforecast(spec, n.ahead=1, n.roll = nr_of_rolls, data = one_serie_window_future, out.sample = nr_of_rolls)

			for (index in 0:nr_of_rolls) {
				forecasts[d+index+1] = fitted(fore)[index+1]
				sigmas[d+index+1] = sigma(fore)[index+1]
			}
			d = d + nr_of_rolls
		}


		
	}
	forecasts_for_all_series[serie_nr,] = forecasts
	sigmas_for_all_series[serie_nr,] = sigmas
	cat(".")

}


forecasts_for_all_series <- mapply(forecasts_for_all_series, FUN=as.numeric)
sigmas_for_all_series <- mapply(sigmas_for_all_series, FUN=as.numeric)
forecasts_for_all_series <- matrix(data=forecasts_for_all_series, ncol=foreLength+1, nrow=nr_of_series)
sigmas_for_all_series <- matrix(data=sigmas_for_all_series, ncol=foreLength+1, nrow=nr_of_series)


file_name_mean = paste("forecasts_mean_", datatype, "_", size, ".csv", sep="")
write.table(forecasts_for_all_series, row.names=FALSE, col.names=FALSE, file = file_name_mean)
file_name_var = paste("forecasts_variance_", datatype, "_", size, ".csv", sep="")
write.table(sigmas_for_all_series, row.names=FALSE, col.names=FALSE, file = file_name_var)
print("End of R Program")