---
  title: "Thera Bank Project"
output: html_document
---
  1. ###EDA###
##Importing the Libraries##

library(readr)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(lattice)
library(DataExplorer)
library(grDevices)
library(factoextra)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ranger)
library(Metrics)
library(ROCit)
library(kableExtra)
library(fpc)
library(NbClust)
library(e1071)
######################

setwd("F:/GREAT LEARNING/DATA MINING/Project -Thera Bank Dataset/GL - Solution")
getwd()
bank=read.csv("Thera Bank_dataset.csv", header = TRUE)
str(bank)
summary(bank)

data.frame(bank)


##Checking the Dimensions##

dim(bank)
##Checking for missing valves##

any(is.na(bank))


sum(is.na(bank))

##checking columns which have missing values##
sapply(bank,function(x)sum(is.na(x)))

## check again after replacing with zero##
bank[is.na(bank)]=0
any(is.na(bank))

dim(bank)

str(bank)

summary(bank)

## removing ID and Zipcode column from dataset##
bank=bank[,-c(1,5)]

## Converting multiple coloumns into Factor columns##
col=c("Education","Personal.Loan","Securities.Account","CD.Account","Online","CreditCard")
bank[col]=lapply(bank[col],factor)

## converting Education into ordered factors .ordinal variable##
bank$Education=factor(bank$Education,levels= c("1","2","3"),order=TRUE)

##Changing the name of few variables for ease of use##
bank = bank%>% rename(Age = "Age..in.years.", Experience ="Experience..in.years.",
                      Income = "Income..in.K.year.")

## checking for rows having negative valves as Experience##

head(bank[bank$Experience<0,])

##Fixing them up##
bank$Experience= abs(bank$Experience)
dim(bank)


summary(bank)

##Exploratory Data Analysis##

##introducting the dataset##
plot_intro(bank)

##Histogram Distributions of Dataset##
##Plotting the histogram for all numerical variables##
plot_histogram(bank)


##Plotting density plot for all numerical variables##
plot_density(bank,geom_density_args = list(fill="blue",alpha=0.4))


##Plotting boxplot by factor of Education for all the numerical variables##
plot_boxplot(bank, by="Education",
             geom_boxplot_args = list("outlier.color"="red"))

##Plotting boxplot for Personal Loan (Response variable) for all numerical variables##
plot_boxplot(bank, by="Personal.Loan",geom_boxplot_args = list("outlier.color"="green"))



View(bank)

##Plotting GGPlot for all variables##
p1 = ggplot(bank, aes(Income, fill= Personal.Loan)) + geom_density(alpha=0.4)
p2 = ggplot(bank, aes(Mortgage, fill= Personal.Loan)) + geom_density(alpha=0.4)
p3 = ggplot(bank, aes(Age, fill= Personal.Loan)) + geom_density(alpha=0.4)
p4 = ggplot(bank, aes(Experience, fill= Personal.Loan)) + geom_density(alpha=0.4)
p5 = ggplot(bank, aes(Income, fill= Education)) + geom_histogram(alpha=0.4, bins = 70)
p6 = ggplot(bank, aes(Income, Mortgage, color = Personal.Loan)) + 
  geom_point(alpha = 0.7)
grid.arrange(p1, p2, p3, p4, p5, p6, ncol = 2, nrow = 3)

## GGplot Education##
ggplot(bank, aes(Education,fill= Personal.Loan)) + 
  geom_bar(stat = "count", position = "dodge") +
  geom_label(stat = "count", aes(label= ..count..), 
             size = 3, position = position_dodge(width = 0.9), vjust=-0.15)+
  scale_fill_manual("Personal Loan", values = c("0" = "pink", "1" = "blue"))+
  theme_minimal()

##GGPlot for Creditcard##
ggplot(bank, aes(Income,y = CCAvg, color = Personal.Loan)) + 
  geom_point(size = 1)

##GGplot for Mortgage##
ggplot(bank, aes(Income,y = Mortgage, color = Personal.Loan)) + 
  geom_point(size = 1)


###Clustering###

bank.clus = bank %>% select_if(is.numeric)

bank.scaled = scale(bank.clus, center = TRUE)

bank.dist = dist(bank.scaled, method = "euclidean")
##checking optimal number of clusters to categorize dataset##


p12 = fviz_nbclust(bank.scaled, kmeans, method = "silhouette", k.max = 5)
p21 = fviz_nbclust(bank.scaled, kmeans, method = "wss", k.max = 5)

grid.arrange(p12, p21, ncol=2)



set.seed(8787)
bank.clusters = kmeans(bank.scaled, 3, nstart = 10)

fviz_cluster(bank.clusters, bank.scaled, geom = "point", 
             ellipse = TRUE, pointsize = 0.2, ) + theme_minimal()


###Splitting of Dataset into Train - Testset###

set.seed(1233)
##sampling 70% of data for training the algorithms using random sampling##

bank.index=sample(1:nrow(bank),nrow(bank)*0.70)
bank.train=bank[bank.index,]
bank.test=bank[-bank.index,]
dim(bank.test)

dim(bank.train)


##checking the ration of persoanl loans in each partition##
table(bank.train$Personal.Loan)
table(bank.test$Personal.Loan)


###cart Model###

set.seed(233)

cart.model.gini= rpart(Personal.Loan~., data=bank.train,method = "class",
                       parms = list(split="gini"))

##checking the complexity parameter##

plotcp(cart.model.gini)


###Plotting the classification Tree###

rpart.plot(cart.model.gini,cex = 0.6)


##checking the cptable to gauge the best crossvalidated error and corresponding
##complexity paramter

cart.model.gini$cptable


##checking for the variable importance for splitting of tree

cart.model.gini$variable.importance


##pruned Cart Tree##
##prunning the tree using the best complexity parameter##

pruned.model=prune(cart.model.gini,cp=0.015)

##plotting the prunned tree##

rpart.plot(pruned.model,cex=0.65)


##Cart Prediction##
cart.pred=predict(pruned.model,bank.test,type="prob")
cart.pred.prob.1=cart.pred[,1]
head(cart.pred.prob.1,10)


##setting the threshold for probabilities to be considered as 1##

threshold = 0.70

bank.test$Loanprediction = ifelse(cart.pred.prob.1 >= threshold, 1, 0)

bank.test$Loanprediction = as.factor(bank.test$Loanprediction)

Cart.Confusion.Matrix =confusionMatrix(bank.test$Loanprediction,
                                       reference = bank.test$Personal.Loan, positive = "1")
Cart.Confusion.Matrix

##Random Forest Model##

set.seed(1233)

RandomForest.model = randomForest(Personal.Loan~., data = bank.train)
print(RandomForest.model)

##print the error rate##

err=RandomForest.model$err.rate
head(err)


##out of bag error##

oob_err=err[nrow(err),"OOB"]
print(oob_err)


##plot the OOB Error##

plot(RandomForest.model)
legend(x="topright",legend = colnames(err),fill=1:ncol(err))

##prediction for Random Forest package##

ranfost.pred = predict(RandomForest.model, bank.test, type = "prob")[,1]

bank.test$RFpred = ifelse(ranfost.pred>=0.8,"1","0")

bank.test$RFpred = as.factor(bank.test$RFpred)

levels(bank.test$RFpred)


RFConf.Matx = confusionMatrix(bank.test$RFpred, bank.test$Personal.Loan, positive = "1")
RFConf.Matx

table(bank.test$Personal.Loan)

##Tuning the Random Forest algo##

set.seed(333)

tuned.RandFors = tuneRF(x = subset(bank.train, select = -Personal.Loan),
                        y= bank.train$Personal.Loan, 
                        ntreeTry = 501, doBest = T)


print(tuned.RandFors)

##Modelling using ranger package##
set.seed(999)

RG.model = train(Personal.Loan~., data = bank.train, tuneLength = 3,
                 method = "ranger", 
                 trControl= trainControl(method = 'cv',
                                         number = 5,
                                         verboseIter = FALSE))
RG.model

plot(RG.model)

##Tuning ranger grid##

tuneGrid = data.frame(.mtry = c(2,4,8), .splitrule = "gini", .min.node.size=1)

set.seed(22222)
RFgrid.model = train(Personal.Loan~., data = bank.train, tuneGrid = tuneGrid,
                     method = "ranger", 
                     trControl= trainControl(method = 'cv',
                                             number = 5, verboseIter = FALSE))
RFgrid.model

plot(RFgrid.model)

##Refind ranger model##

set.seed(101)

Range.model = ranger(Personal.Loan~., data = bank.train, num.trees = 511,
                     mtry = 4, min.node.size = 1, verbose = FALSE)

Range.model


##Prediction of RangeR package##

range.pred = predict(Range.model, bank.test)

table(bank.test$Personal.Loan, range.pred$predictions)


##Confusion Matrix of RangeR Package##

Range.ConMatx = confusionMatrix(range.pred$predictions, 
                                bank.test$Personal.Loan, positive = "1")
Range.ConMatx

##Plotting of ROC Curve##

Prediction.Labels = as.numeric(range.pred$predictions)
Actual.Labels = as.numeric(bank.test$Personal.Loan)

roc_Rf = rocit(score = Prediction.Labels, class = Actual.Labels)

plot(roc_Rf)


