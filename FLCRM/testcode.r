rm(list=ls())
library(locpol)
library(MASS)
library(survival)
library(survAUC)
library(survcomp)



# sample size
n<-1000
# repetion
nrep<-500

load("newsimulationpoweroptimaltauC1.dat")

# measurement error variance
measurementerrorvariance<-0.5

censoringindex<-3

censoringvalues<-c(0.1, 0.3, 0.5)

##### censoring rate
censoring<-censoringvalues[censoringindex]

h<-calh<-0.01

p<-1/h+1
calp<-p

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
betafun<-function(C1, C2, t) {
	sigma<-0.3
	betafun<-0.3*(C1*(sin(pi*t)-cos(pi*t)+sin(3*pi*t)-cos(3*pi*t)+sin(5*pi*t)/9-cos(5*pi*t)/9+sin(7*pi*t)/16-cos(7*pi*t)/16+sin(9*pi*t)/25-cos(9*pi*t)/25)+C2*(1/sqrt(2*pi)/sigma)*exp(-(t-0.5)^2/2/sigma^2))}

q<-4

gamma<-rep(0.2,q)

correlationmatrix<-matrix(0,nrow(Sigma),q)
correlationmatrix[1,1:q]<-0.1
correlationmatrix[1:q,1]<-0.1

SigmaZ<-0.5^t(sapply(1:q, function(i, j) abs(i-j), 1:q))

Bigsigma<-rbind(cbind(Sigma, correlationmatrix), cbind(t(correlationmatrix), SigmaZ))

mu<-rep(0,nrow(Bigsigma))

cvalue<-seq(0.1,1,0.1)

thresholdvalues<-c(0.70, 0.75, 0.80, 0.85, 0.90, 0.95)

testresult<-array(NA, c(nrep, length(cvalue), length(thresholdvalues)))

for (s in 1:nrep){
set.seed(2014+10000*s)

data<-mvrnorm(n,mu,Bigsigma)

ximatrix<-data[,1:nrow(Sigma)]

rawX<-ximatrix%*%tempbasematrix

calX<-rawX

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

scaleX<-scale(X,center=TRUE,scale=FALSE)
Eigvec=svd(scaleX)$v
Singval=svd(scaleX)$d
est_Eigval=Singval^2*h/(n-1)


for (ii in 1:length(cvalue)){
C1<-cvalue[ii]

tau<-optimaltau[ii+1,censoringindex]

beta<-rep(0,calp)
for (jj in 1:calp) {
beta[jj]<-betafun(C1,0, designpoints[jj])
}

truebeta<-beta

###### generate the time-to-event data
parameter<-exp(calX%*%beta*calh+Z%*%gamma)

failuretime<-rexp(n, rate = parameter)
censoringtime<-runif(n,0,tau)

event<-rep(0,n)
event<-as.numeric(failuretime<censoringtime)

time<-failuretime*event+censoringtime*(rep(1,n)-event)

for (j in 1:length(thresholdvalues)){
threshold<-thresholdvalues[j]

for (ss in 1:p) {
if (sum(est_Eigval[1:ss])/sum(est_Eigval)>threshold)
break
}

selectindex<-ss

est_Eigfun=Eigvec[,1:selectindex]/sqrt(h)


eigenscore<-scaleX%*%est_Eigfun*h

designmatrix<-cbind(eigenscore,Z)

nullcoxresult<-coxph(formula=Surv(time,event)~Z)

initial.value<-c(rep(0,selectindex),nullcoxresult$coef)

fullcoxresult<-coxph(formula=Surv(time,event)~designmatrix, init=initial.value)

testscore<-fullcoxresult$score

criticalvalue<-1-pchisq(testscore, selectindex, lower.tail = TRUE, log.p = FALSE)

testresult[s,ii, j]<-as.numeric(criticalvalue<0.05)
}
}
}

power<-apply(testresult,c(2,3),mean)

save(power, file=paste("newsimulationFVEpower_n", n, "_c", 10*censoring, "_me", 10*measurementerrorvariance, "cindexC1.dat", sep=""))







