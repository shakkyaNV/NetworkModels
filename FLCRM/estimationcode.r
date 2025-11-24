rm(list=ls())
library(locpol)
library(MASS)
library(survival)
library(survAUC)
library(survcomp)
#library(pcox)

n<-1000

# measurementerror variance
measurementerrorvariance<-0.5

censoringindex<-3

censoringvalues<-c(0.1, 0.3, 0.5)

load("newsimulationcensoring.dat")

tau<-threshold[censoringindex]

censorrate<-censoringvalues[censoringindex]


h<-calh<-0.01

p<-1/h+1

designpoints<-seq(0,1,calh)
tempbasematrix<-matrix(NA,22,length(designpoints))
for (j in 1:10){
	tempbasematrix[j,]<-sqrt(2)*sin(pi*(2*j-1)*designpoints)
}
for (j in 11:20){
	tempbasematrix[j,]<-sqrt(2)*cos(pi*(2*j-21)*designpoints)
}

tempbasematrix[21,]<-1
tempbasematrix[22,]<-designpoints

Sigma<-matrix(0,22,22)
Sigma[1,1]<-1
Sigma[11,11]<-1
for (j in 2:10){
Sigma[j,j]<-(j-1)^(-2)
}
for (j in 12:20){
Sigma[j,j]<-(j-11)^(-2)	
}

Sigma[21,21]<-1
Sigma[22,22]<-1


# betafunction
betafun<-function(t) {
	sigma<-0.3
	betafun<-0.3*(sin(pi*t)-cos(pi*t)+sin(3*pi*t)-cos(3*pi*t)+sin(5*pi*t)/9-cos(5*pi*t)/9+sin(7*pi*t)/16-cos(7*pi*t)/16+sin(9*pi*t)/25-cos(9*pi*t)/25+(1/sqrt(2*pi)/sigma)*exp(-(t-0.5)^2/2/sigma^2))
}

beta<-betafun(designpoints)

truebeta<-beta

q<-4

gamma<-rep(0.2,q)

correlationmatrix<-matrix(0,nrow(Sigma),q)
correlationmatrix[1,1:q]<-0.1
correlationmatrix[1:q,1]<-0.1

SigmaZ<-0.5^t(sapply(1:q, function(i, j) abs(i-j), 1:q))

Bigsigma<-rbind(cbind(Sigma, correlationmatrix), cbind(t(correlationmatrix), SigmaZ))

mu<-rep(0,nrow(Bigsigma))

hh<-8

set.seed(2014)

data<-mvrnorm(n,mu,Bigsigma)

ximatrix<-data[,1:nrow(Sigma)]

rawX<-ximatrix%*%tempbasematrix

options(warn=-1)

measurementerror<-matrix(rnorm(n*p,0,sqrt(measurementerrorvariance)),n,p)
observeX<-rawX+measurementerror

X<-matrix(NA,n,p)
for (i in 1:n){
dataframe <- data.frame(designpoints)
dataframe$observeX<-observeX[i,]
lpfit <- locpol(observeX~designpoints,dataframe, xeval=designpoints)
X[i,]<-lpfit$lpFit[,2]
}

Z<-data[,(nrow(Sigma)+1):(nrow(Sigma)+q)]

###### generate the time-to-event data
parameter<-exp(rawX%*%beta*calh+Z%*%gamma)

failuretime<-rexp(n, rate = parameter)
censoringtime<-runif(n,0,tau)

event<-rep(0,n)
event<-as.numeric(failuretime<censoringtime)

time<-failuretime*event+censoringtime*(rep(1,n)-event)

scaleX<-scale(X,center=TRUE,scale=FALSE)
Eigvec=svd(scaleX)$v
Singval=svd(scaleX)$d
est_Eigval=Singval^2*h/(n-1)

totaleigenscore<-scaleX%*%(Eigvec/sqrt(h))*h

ordertime<-order(time)
sorttime<-time[ordertime]
sortevent<-event[ordertime]

sortfailureindex<-which(sortevent==1)

#### AIC selection result
AIC<-loglikelihood<-rep(0,hh)
for (ss in 1:hh) {
est_Eigfun=Eigvec[,1:ss]/sqrt(h)

eigenscore<-scaleX%*%est_Eigfun*h

designmatrix<-cbind(eigenscore,Z)

fullcoxresult<-coxph(formula=Surv(time,event)~designmatrix)

orderdesignmatrix<-designmatrix[ordertime,]

estimatecoefficient<-fullcoxresult$coefficients

fullcoxcurve<-survfit(fullcoxresult)

finalresult<-summary(fullcoxresult)

loglikelihood[ss]<-finalresult$loglik[2]


AIC[ss]<-2*ss-2*loglikelihood[ss]
}

AICselectindex<-which.min(AIC)

est_Eigfun=Eigvec[,1:AICselectindex]/sqrt(h)

eigenscore<-scaleX%*%est_Eigfun*h

designmatrix<-cbind(eigenscore,Z)

newdataframe<-data.frame(cbind(time, event, designmatrix))

totalvariable<-"V3"
for (i in 4:(ncol(designmatrix)
+2)){
totalvariable<-paste(totalvariable,"+V", i, sep="")
}

totalformular<-as.formula(paste("Surv(time,event)~", totalvariable, sep=""))

fullcoxresult<-coxph(formula=totalformular, data=newdataframe)

fullcoxcurve<-survfit(fullcoxresult)

finalresult<-summary(fullcoxresult)

summarycoefficient<-finalresult$coefficients

coefficient<-summarycoefficient[,1]
secoefficient<-summarycoefficient[,3]

#### Estimated Beta
AICestimatebeta<-est_Eigfun%*%as.matrix(coefficient[1:AICselectindex])

#### Estimated Gamma
AICestimategamma<-coefficient[-(1:AICselectindex)]


