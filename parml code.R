# This script loads and prepares training data
rm(list = ls())


library("randomForest")
library("caret")
library("pROC")
library("ROCR")
library("plyr")
library("missForest")
library("tidyverse")
library("gbm")
library("pdp")
library("ggplot2")
library("iml")

# load functions
source("functions.R")
source("functionsCV.R")
#------------------------------------------------------------------
########Import data#################
#------------------------------------------------------------------

# ## define column classes
colClasses <- rep("factor",16)
colClasses[c(2:6,10:16)] <- "numeric"

#load data for training
data <- read.csv("CDVagesmp.csv", fill=TRUE,colClasses=colClasses);str(data)
#data cleaning
# 
data$Spatial.coordinates.X <- NULL 
data$Spatial.coordinates.Y <- NULL
str(data) #drop year/lat/long

#------------------------------------------------------------------
################# A. Drop NAs#################
#------------------------------------------------------------------
# 
data<-data[complete.cases(data), ];str(data) #dropping NAs
# ################# B. Imputation with missing forests#################
#------------------------------------------------------------------
# 
#impute missing values with missForest. Default values should work
set.seed(135)
data <- missForest(data, variablewise = FALSE)
#check how successful that was. Normalized root mean square error (NRMSE) for continuos variables, PFC = falsely classified entries for categorical
data$OOBerror

data <- as.data.frame(data$ximp);str(data)

#------------------------------------------------------------------
########Check correlation structure#################
#------------------------------------------------------------------

# CorData <-  data[-which(names(data) == "Class")] #all predictors
# CorDataNum <- select_if(CorData, is.numeric)
# # calculate correlation matrix
# correlationMatrix <- cor(CorDataNum)
# # summarize the correlation matrix
# print(correlationMatrix)
# # find attributes that are highly corrected (ideally >0.75)
# highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# # print indexes of highly correlated attributes
# print(highlyCorrelated)

#------------------------------------------------------------------
########Select only relevent features#################
#------------------------------------------------------------------
#assign class

colnames(data)[1] <- "Class"

X = data[-which(names(data) == "Class")]
Y <- data$Class


library("Boruta")
set.seed(124)
ImpVar <- Boruta(X, Y, doTrace = 0, maxRuns = 500) #, getImp = getImpRfGini seems to be a bit more severe compared to rfZ
print(ImpVar)
#plot with x axis label vertical

plot(ImpVar, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(ImpVar$ImpHistory),function(i)
  ImpVar$ImpHistory[is.finite(ImpVar$ImpHistory[,i]),i])
names(lz) <- colnames(ImpVar$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(ImpVar$ImpHistory), cex.axis = 0.7)

#plotImpHistory(ImpVar) #optional - look at the runs in more detail

set <- getSelectedAttributes(ImpVar, withTentative = TRUE)#keep tentative
dataReduced <- data[which(names(data) %in% set)]
data<- cbind(data[1], dataReduced);str(data)
-----------------------------------------------------
  ########create machine learning objects#################
#------------------------------------------------------------------
colnames(data)[1] <- "Class"
# create two partitions: training and testing data
inTrain= createDataPartition(y=data$Class, p = .8, list = FALSE)

data.train.descr <- data[inTrain,-1]
data.train.class <- data[inTrain,1]
data.test.descr <- data[-inTrain,-1]
data.test.class <- data[-inTrain,1]
#remove if neceassary
rm(colClasses,inTrain)

# ## create folds for CV
set.seed(123)
myFolds <- createMultiFolds(y=data.train.class,k=10,times=10)

#  
# # ## create folds for CV, performing downsample of majority class
set.seed(130)
myFoldsDownSample <- CreateBalancedMultiFolds(y=data.train.class,k=10,times=10) 
length(unique(unlist(myFoldsDownSample)))

####--------------------------------------------------------------
# Optional: do analyses in parallell 
####---------------------------------------------------------------


#library(doParallel) #not necessary
# for parallel model training
#registerDoParallel(cores = 4)

#------------------------------------------------------------------
#################Set up cross validation#################
#------------------------------------------------------------------

#all data - no downsampling

myControl <- trainControl(## 10-fold CV)
  method = "repeatedcv",
  number = 10,
  repeats = 10,
  index = myFolds,
  savePredictions=TRUE,        
  classProbs = TRUE,
  summaryFunction = twoClassSummary,  
  allowParallel=TRUE,
  selectionFunction = "best")

#downsampled version

myControlDownSample <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  repeats = 10,
  index = myFoldsDownSample,
  savePredictions=TRUE,        
  classProbs = TRUE,
  summaryFunction = twoClassSummary,  
  allowParallel=TRUE,
  selectionFunction = "best")

#------------------------------------------------------------------
#################Random Forests #################
#------------------------------------------------------------------


# tune mtry parameters (number of predictors) and perform cross-validation on data
rf.grid <- expand.grid(.mtry=(3:6)) #number of predictors (mtry) to test

#Run model
set.seed(125)
rf.fit <- train(data.train.descr,data.train.class,
                method = "rf",
                metric = "ROC",
                verbose = FALSE,
                trControl = myControlDownSample,
                tuneGrid = rf.grid,
                verboseIter=TRUE
)

plot(rf.fit)

# estimate model performance in terms of a confusion matrix from repeated cross-validation
oof <- rf.fit$pred #01 fold cross validation -check if worked
head(oof)
tail(oof)#mtry = covariates used
# consider only the best tuned model - this checks performance
oof <- oof[oof$mtry==rf.fit$bestTune[,'mtry'],]
repeats<-rf.fit$control$repeats
rf.performance.cv <- EstimatePerformanceCV(oof=oof,repeats=repeats) #Matthew's correlation coefficient (MCC)
rf.performance.cv 
# estimate error rates from repeated cross-validation
rf.error.cv <- EstimateErrorRateCV(oof=oof,repeats=repeats)
rm(oof,repeats)


#save the model
save(rf.fit,file="Parvo Age Sampled RF downsampled.RData")


####variable importance calculation and plots using a permutation method

library("iml")

X = data[-which(names(data) == "Class")]
Y <- data$Class

mod = Predictor$new(rf.fit, data = X, y = Y) #create predictor object

#Variable importance using permutation

imp = FeatureImp$new(mod, loss = "ce", method="cartesian") #cartesian is more thorough but 'shuffle' is faster
imp.dat = imp$results

plot(imp)+ theme_bw()
####Interactions

interact <- Interaction$new(mod)
plot(interact)+ theme_bw()

str(data)
interact <- Interaction$new(mod, feature = "Yearly.Rainfall")

plot(interact)+ theme_bw()
#plot variable with strongest interactions
pdp.obj <- Partial$new(mod, feature = c("Yearly.Rainfall","AvgGroupNumberPrev2year"))
plot(pdp.obj)+ scale_fill_gradient(low = "white", high = "red")

#Tree Summaries
tree = TreeSurrogate$new(mod, maxdepth = 3)
plot(tree)
tree$r.squared
head(tree$results)

#explain single predictions using game theory.

shapley = Shapley$new(mod, x.interest = X[1,])
shapley$plot()+ theme_bw()


results = shapley$results
head(results)



#------------------------------------------------------------------
#################Gradient Boosting #################
#------------------------------------------------------------------
# 
# #set up GBM tuning paramters
gbm.grid <-  expand.grid(interaction.depth = c(1,3,5,7,9),
                         n.trees = (1:30)*10,
                         shrinkage = 0.1,
                         n.minobsinnode = c(10))# will stop when is 10 onservation in terminal node

nrow(gbm.grid)
set.seed(123)
gbm.fit <- train(data.train.descr, data.train.class,
                 method = "gbm",
                 metric = "ROC",
                 verbose = FALSE,
                 trControl = myControl,
                 ## Now specify the exact models 
                 ## to evaludate:
                 tuneGrid = gbm.grid)
# 
plot(gbm.fit)
save(gbm.fit,file="CDV Age Sampled GBMAll.RData")
# # estimate model performance in terms of a confusion matrix from repeated cross-validation
oof <- gbm.fit$pred
# # consider only the best tuned model
oof <- oof[intersect(which(oof$n.trees==gbm.fit$bestTune[,'n.trees']),which(oof$interaction.depth==gbm.fit$bestTune[,'interaction.depth'])),]
repeats <- gbm.fit$control$repeats
gbm.performance.cv <- EstimatePerformanceCV(oof=oof,repeats=repeats)
gbm.performance.cv
# # estimate error rates from repeated cross-validation
gbm.error.cv <- EstimateErrorRateCV(oof=oof,repeats=repeats)
gbm.error.cv 
# ####variable importance calculation and plots using a permutation method
# 
library("iml")
# 
X <-data[-which(names(data) == "Class")]
Y <- data$Class
# 
mod <-Predictor$new(gbm.fit, data = X, y = Y) #create predictor object
imp <-FeatureImp$new(mod, loss = "ce", method="cartesian") #cartesian is more thorough but 'shuffle' is faster
imp.dat<- imp$results
plot(imp)+ theme_bw()
# 
####Interactions

interact <- Interaction$new(mod)
plot(interact)

#look at the top interacting predictor
str(data)
interact <- Interaction$new(mod, feature = "FIV_PCoA1")

plot(interact)

pdp.obj <- Partial$new(mod, feature = c("FIV_PCoA1","territoryOverlap"))

plot(pdp.obj)+ scale_fill_gradient(low = "white", high = "red")


# pdp.obj = Partial$new(mod, feature = c("agesmp", "habitatQuality"))
# plot(pdp.obj)+ scale_fill_gradient(low = "white", high = "red")
# 
# pdp.obj = Partial$new(mod, feature = c("agesmp", "AverageNumberNeighbours"))
# plot(pdp.obj)+ scale_fill_gradient(low = "white", high = "red")
# 

## summary data for center to the minimum
summary(data)

#partial dependency plots
pd = Partial$new(mod, feature ="territoryOverlap", ice = TRUE, aggregation = "pdp", 
                 grid.size = 20, center.at = 0, run = TRUE)
plot(pd)+ theme_bw()

pd = Partial$new(mod, feature ="prideTerritorySize", ice = TRUE, aggregation = "pdp", 
                 grid.size = 20, center.at = 19.98,  run = TRUE)
plot(pd)+ theme_bw()

pd = Partial$new(mod, feature ="AverageNumberNeighbours", ice = TRUE, aggregation = "pdp", 
                 grid.size = 20, center.at = 0.00,  run = TRUE)
plot(pd)+ theme_bw()

pd = Partial$new(mod, feature ="agesmp", ice = TRUE, aggregation = "pdp", 
                 grid.size = 20, center.at = 1.0 , run = TRUE)
plot(pd)+ theme_bw()

pd = Partial$new(mod, feature ="Sex", ice = TRUE, aggregation = "pdp", 
                 grid.size = 20, run = TRUE)
plot(pd)+ theme_bw()
pd = Partial$new(mod, feature ="Prides", ice = TRUE, aggregation = "pdp", 
                 grid.size = 20, run = TRUE)
plot(pd)+ theme_bw()

pd = Partial$new(mod, feature ="Despotic", ice = TRUE, aggregation = "pdp", 
                 grid.size = 20, run = TRUE)
plot(pd)+ theme_bw()


pd = Partial$new(mod, feature ="Yearly.Rainfall", ice = TRUE, aggregation = "pdp", 
                 grid.size = 20, center.at =  19.26, run = TRUE)
plot(pd)+ theme_bw()


pd = Partial$new(mod, feature ="Average.Ph", ice = TRUE, aggregation = "pdp", 
                 grid.size = 20, center.at =  6.8, run = TRUE)
plot(pd)+ theme_bw()


pd = Partial$new(mod, feature ="Median.vegetation.cover....", ice = TRUE, aggregation = "pdp", 
                 grid.size = 20, center.at =  17.30, run = TRUE)
plot(pd)+ theme_bw()
pd = Partial$new(mod, feature ="habitatQuality", ice = TRUE, aggregation = "pdp", 
                 grid.size = 20, center.at =  26.27, run = TRUE)
plot(pd)+ theme_bw()


pd = Partial$new(mod, feature ="AvgGroupNumberPrev2year", ice = TRUE, aggregation = "pdp", 
                 grid.size = 20, center.at =  0.000, run = TRUE)
plot(pd)+ theme_bw()

#pd = Partial$new(mod, feature ="Years_since_epidemic", ice = TRUE, aggregation = "pdp", 
grid.size = 20, center.at =  0.000, run = TRUE)
#plot(pd)+ theme_bw()


#pd = Partial$new(mod, feature ="Age_exposed", ice = TRUE, aggregation = "pdp", 
grid.size = 20, center.at =  0.000, run = TRUE)
#plot(pd)+ theme_bw()

#FIV doesn't make sense to center
pd = Partial$new(mod, feature ="FIV_PCoA1", ice = TRUE, aggregation = "pdp", 
                 grid.size = 20, run = TRUE)
plot(pd)+ theme_bw()

pd = Partial$new(mod, feature ="FIV_PCoA2", ice = TRUE, aggregation = "pdp", 
                 grid.size = 20, run = TRUE)
plot(pd)+ theme_bw()



####tree summaries

tree <- TreeSurrogate$new(mod, maxdepth = 4)
plot(tree)
tree$r.squared
head(tree$results)


#explain single predictions using game theory.

shapley = Shapley$new(mod, x.interest = X[1,])
shapley$plot()

results = shapley$results
head(results)



#------------------------------------------------------------------
#################Support vector machine #################
#------------------------------------------------------------------
# 
# 
set.seed(123)
svm.fit <- train(Class ~ .,data=cbind(Class=data.train.class,data.train.descr),
                 method = "svmRadial",
                 tuneLength = 9,
                 metric="ROC",
                 trControl = myControlDownSample)
plot(svm.fit)
# 
# # estimate model performance in terms of a confusion matrix from repeated cross-validation
oof <- svm.fit$pred
# # consider only the best tuned model
oof <- oof[intersect(which(oof$sigma==svm.fit$bestTune[,'sigma']),which(oof$C==svm.fit$bestTune[,'C'])),]
repeats <- svm.fit$control$repeats
svm.performance.cv <- EstimatePerformanceCV(oof=oof,repeats=repeats)
svm.performance.cv 
# # estimate error rates from repeated cross-validation
svm.error.cv <- EstimateErrorRateCV(oof=oof,repeats=repeats)
rm(oof,repeats)
# save the model
save(svm.fit, file="CDVAgeSampledSVM_down.RData")

# 
# 
# #######variable importance
# X = data[-which(names(data) == "Class")]
# Y <- data$Class
# 
# mod = Predictor$new(svm.fit, data = X, y = Y) #create predictor object
# imp = FeatureImp$new(mod, loss = "ce", method="cartesian")
# imp.dat = imp$results
# plot(imp)+
#   theme_bw()
# 
# 
#-------------------------------------------------------------------------------
#TUNE AND TRAIN LOGISTIC REGRESSION MODEL


set.seed(155)
glm.fit <- train(data.train.descr,data.train.class,
                 method = "glm",
                 metric="ROC", # will use accuracy instead
                 family="binomial",
                 trControl=myControlDownSample)


# estimate model performance in terms of a confusion matrix from repeated cross-validation
oof <- glm.fit$pred
# consider only the best tuned model
oof <- oof[which(oof$parameter==glm.fit$bestTune[,'parameter']),]
repeats<-glm.fit$control$repeats
glm.performance.cv <- EstimatePerformanceCV(oof=oof,repeats=repeats)
glm.performance.cv
# estimate error rates from repeated cross-validation
glm.error.cv <- EstimateErrorRateCV(oof=oof,repeats=repeats)
rm(oof,repeats)


save(glm.fit,glm.performance.cv,glm.error.cv,file="model_glm_fullfeatset_Atb.RData")

# 
# ####variable importance calculation and plots using a permutation method



X = data[-which(names(data) == "Class")]
Y <- data$Class

mod = Predictor$new(glm.fit, data = X, y = Y) #create predictor object

#Variable importance using permutation

imp = FeatureImp$new(mod, loss = "ce", method="cartesian") #cartesian is more thorough but 'shuffle' is faster
imp.dat = imp$results

plot(imp)+ theme_bw()
# 
# 
# 
