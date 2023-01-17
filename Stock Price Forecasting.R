#Stock market trend forcasting
#Project for CS512 Machine Learning in Medicine and Health
#By Edin Ziga 190302192
#⠀⠀⠀⠀⠀⠀⠀   ⠀⣠⣤⣤⣤⣤⣤⣶⣦⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀ 
#⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⡿⠛⠉⠙⠛⠛⠛⠛⠻⢿⣿⣷⣤⡀⠀⠀⠀⠀⠀ 
#⠀⠀⠀⠀⠀⠀⠀⠀⣼⣿⠋⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⠈⢻⣿⣿⡄⠀⠀⠀⠀ 
#⠀⠀⠀⠀⠀⠀⠀⣸⣿⡏⠀⠀⠀⣠⣶⣾⣿⣿⣿⠿⠿⠿⢿⣿⣿⣿⣄⠀⠀⠀ 
#⠀⠀⠀⠀⠀⠀⠀⣿⣿⠁⠀⠀⢰⣿⣿⣯⠁⠀⠀⠀⠀⠀⠀⠀⠈⠙⢿⣷⡄⠀ 
#⠀⠀⣀⣤⣴⣶⣶⣿⡟⠀ ⠀⢸⣿⣿⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣷⠀ 
#⠀⢰⣿⡟⠋⠉⣹⣿⡇⠀⠀⠘⣿⣿⣿⣿⣷⣦⣤⣤⣤⣶⣶⣶⣶⣿⣿⣿⠀ 
#⠀⢸⣿⡇⠀⠀⣿⣿⡇⠀⠀⠀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠃⠀ 
#⠀⣸⣿⡇⠀⠀⣿⣿⡇⠀ ⠀⠀⠉⠻⠿⣿⣿⣿⣿⡿⠿⠿⠛⢻⣿⡇⠀⠀ 
#⠀⣿⣿⠁⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⢸⣿⣧⠀⠀ 
#⠀⣿⣿⠀⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⢸⣿⣿⠀⠀ 
#⠀⣿⣿⠀⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⢸⣿⣿⠀⠀ 
#⠀⢿⣿⡆⠀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⢸⣿⡇⠀⠀ 
#⠀⠸⣿⣧⡀⠀⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⣿⣿⠃⠀⠀ 
#⠀⠀⠛⢿⣿⣿⣿⣿⣇⠀ ⠀⠀⣰⣿⣿⣷⣶⣶⣶⣶⠶⠀ ⢠⣿⣿ ⠀⠀⠀ 
#⠀⠀⠀⠀⠀⠀⠀⣿⣿⠀⠀⠀⠀⠀⣿⣿⡇⠀⣽⣿⡏⠁⠀⠀ ⢸⣿⡇⠀⠀⠀ 
#⠀⠀⠀⠀⠀⠀⠀⣿⣿⠀⠀⠀⠀⠀⣿⣿⡇⠀⢹⣿⡆⠀⠀⠀ ⣸⣿⠇⠀⠀⠀ 
#⠀⠀⠀⠀⠀⠀⠀⢿⣿⣦⣄⣀⣠⣴⣿⣿⠁⠀⠈⠻⣿⣿⣿⣿⡿⠏⠀⠀⠀⠀ 
#⠀⠀⠀⠀⠀⠀⠀⠈⠛⠻⠿⠿⠿⠿⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀

#Libraries
#-------------------------------------------------------------------------------
library(tseries)
library(DataCombine)
library(tsfknn)
library(forecast)
library(xts)
library(fBasics)
library(FinTS)
library(quantmod)
library(rugarch)
library(caTools)

#For the project the following concept was applied
#Stock data was taken for the period between 1-1-2010 and 01-01-2015
#All models were applied for periods of 7,14,30,90,180,252 days
#Models were applied for three stocks, ^GSPC, AMZN, META and TSLA


#Data Analysis - ADF Test for each stock
#-------------------------------------------------------------------------------
rm(list=ls())

getSymbols("^GSPC",src="yahoo",from="2010-01-01",to = "2015-01-01")
inputData <- GSPC$GSPC.Close
plot(GSPC$GSPC.Close)

#Augmented Dickey-Fuller Test
adf.test(inputData)

getSymbols("AMZN",src="yahoo",from="2010-01-01",to = "2015-01-01")
inputData <- AMZN$AMZN.Close
plot(AMZN$AMZN.Close)

adf.test(inputData)


getSymbols("META",src="yahoo",from="2010-01-01",to = "2015-01-01")
inputData <- META$META.Close
plot(META$META.Close)

adf.test(inputData)

getSymbols("TSLA",src="yahoo",from="2010-01-01",to = "2015-01-01")
inputData <- TSLA$TSLA.Close
plot(TSLA$TSLA.Close)

adf.test(inputData)

#ARIMA Model
#-------------------------------------------------------------------------------
rm(list=ls())

getSymbols("TSLA",src="yahoo",from="2010-01-01",to = "2016-01-01")
inputData <- TSLA$TSLA.Close
plot(inputData)

len=length(inputData)-252
#Peridction period varuable
predict=252

input_train = inputData[1:len, ]
input_test = inputData[len:(len+predict),]

#Model based on train period
arima_model <- auto.arima(input_train,stationary=FALSE,lambda="auto")

summary(arima_model)

df = data.frame(day=len:(len+predict), coredata(input_test))

#Forecast
test1 <- forecast(arima_model, h=predict)


#Accuracy and correlation data
accuracy(test1, input_test[1:predict], test=NULL, d=NULL, D=NULL)

cor(test1$lower,inputData[1:predict])
cor(test1$mean,inputData[1:predict])
cor(test1$upper,inputData[1:predict])


#For Historic plotting
plot(test1,xlab = predict, col="red", lwd = 3)

lines(df, col="red", lwd = 3)
grid(nx = NULL, ny = NULL,
     lty = 2,      # Grid line type
     col = "gray", # Grid line color
     lwd = 2)      # Grid line width

#For detailed plotting
plot(test1,xlab = predict, xlim=c(1100,1400), ylim=c(9,60), col="red", lwd = 3)

lines(df, col="red", lwd = 3)
grid(nx = NULL, ny = NULL,
     lty = 2,      # Grid line type
     col = "gray", # Grid line color
     lwd = 2)      # Grid line width


#KNN Regression
#-------------------------------------------------------------------------------
rm(list=ls())

getSymbols("TSLA",src="yahoo",from="2010-01-01",to = "2016-01-01")

inputData <- TSLA$TSLA.Close
len=length(inputData)-252
#Prediction period varable
predict=252

input_train = inputData[1:len, ]
input_test = inputData[len:(len+predict),]

dfinput_Train <- data.frame(ds = index(input_train),
                 y = as.numeric(input_train[,1]))

#Finding the proper k and lag based on test period
k=as.integer(sqrt(length(input_test))/2)
if(k%%2==0) k=k+1
if(k<3) k=3
lagPeriod=as.integer(0.11*length(input_test))
if(lagPeriod<1) lagPeriod=1
predknn <- knn_forecasting(dfinput_Train$y, h =length(input_test), lags = 1:lagPeriod, k = k, msas = "MIMO")
predknn


dfInputDataIndexed <- data.frame(ds = index(1:(len+252)),
                            y = as.numeric(inputData[,1]))

dfInputDataIndexedShortened <- dfInputDataIndexed[1:(len+predict+1), ]

#Accuracy and correlation
accuracy(predknn$prediction, input_test[1:predict])
cor(predknn$prediction,input_test)

print(k)
print(lagPeriod)

#Historic plotting
plot(dfInputDataIndexedShortened, type ="l",col="blue",lwd = 2, pch=1,ylim=c(0,25))
lines(predknn$prediction, col="red", lwd = 3)
grid(nx = NULL, ny = NULL,
     lty = 2,      # Grid line type
     col = "gray", # Grid line color
     lwd = 2)      # Grid line wid

#Detailed Plotting
plot(dfInputDataIndexedShortened, type ="l",col="blue", xlim=c(1130,1400), ylim=c(12,25),lwd = 2, pch=1, xlab=k)
lines(predknn$prediction, col="red", lwd = 3)
grid(nx = NULL, ny = NULL,
     lty = 2,      # Grid line type
     col = "gray", # Grid line color
     lwd = 2)      # Grid line width


#Feed-Forward Neural Network
#--------------------------------------------------------------------------------
rm(list=ls())
getSymbols("TSLA",src="yahoo",from="2010-01-01",to = "2016-01-01")
inputData <- TSLA$TSLA.Close

set.seed("123")
len=length(inputData)-252
#Prediction period
predict=252

input_train = inputData[1:len, ]
input_test = inputData[len:(len+predict),]

#Network config and model
lambda = BoxCox.lambda(inputData$TSLA.Close)
nn = nnetar(input_train,lambda=lambda)
nn

#Forecast
#DO NOT RUN, takes forever to finish
test1 = forecast(nn,h=predict, PT=T)

df = data.frame(day=len:(len+predict), coredata(input_test))


#Accuracy and correlation testing
accuracy(test1, input_test[1:predict], test=NULL, d=NULL, D=NULL)
cor(test1$lower,inputData[1:predict])
cor(test1$mean,inputData[1:predict])
cor(test1$upper,inputData[1:predict])

#For Historic plotting
plot(test1,xlab = predict, col="blue",lwd = 2)
lines(df, col="red", lwd = 2)
grid(nx = NULL, ny = NULL,
     lty = 2,      # Grid line type
     col = "gray", # Grid line color
     lwd = 2)      # Grid line width


#For detailed plotting
plot(test1,xlab = predict, xlim=c(1100,1400), ylim=c(12,19), col="red", lwd = 3)
lines(df, col="red", lwd = 3)
grid(nx = NULL, ny = NULL,
     lty = 2,      # Grid line type
     col = "gray", # Grid line color
     lwd = 2)      # Grid line width


