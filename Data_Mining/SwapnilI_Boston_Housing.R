library(MASS)
#lm####
data(Boston)
View(Boston)
set.seed(10743959)
subset=sample(nrow(Boston),nrow(Boston)*0.75)
train=Boston[subset,]
test=Boston[-subset,]
str(Boston)
summary(Boston)
cor(Boston)
dim(train)
dim(test)
plot(train)
library(leaps)
nullmodel=lm(medv~1,data=train)
fullmodel=lm(medv~.,data=train)
summary(nullmodel)
summary(fullmodel)
model.step=step(nullmodel,scope=list(lower=nullmodel,upper=fullmodel),direction='both')
model.step.summary=summary(model.step)
model.step.summary
#in sample####
(model.step.summary$sigma)^2
(model.step.summary$adj.r.squared)
AIC(model.step)
BIC(model.step)
#Out sample
pi=predict(model.step,test)
mean((pi-test$medv)^2)
#GLM

#CARt####
library(rpart)
boston.rpart<-rpart(medv~.,train)
plot(boston.rpart)
text(boston.rpart)
#insample
boston.train.pred.tree=predict(boston.rpart)
mean((boston.train.pred.tree-train$medv)^2)
#outsample
boston.test.pred.tree=predict(boston.rpart,test)
mean((boston.test.pred.tree-test$medv)^2)
#cart with pruning
boston.largetree <- rpart(formula = medv ~ ., data = train, cp = 0.001)
plot(boston.largetree)
plotcp(boston.largetree)
printcp(boston.largetree)
#insample
boston.train.pred.tree.cp=predict(boston.largetree)
mean((boston.train.pred.tree.cp-train$medv)^2)
#outsample
boston.test.pred.tree.cp=predict(boston.largetree,test)
mean((boston.test.pred.tree.cp-test$medv)^2)




#GAM####
install.packages("mgcv")
library(mgcv)
Boston.gam <- gam(medv ~
                    s(crim)+s(zn)+s(indus)+chas+s(nox)+s(rm)+s(age)
                  +s(dis)+rad+s(tax)+s(ptratio)+s(black)+s(lstat),
                  data=train)
summary(Boston.gam)
plot(Boston.gam, shade = TRUE, , seWithMean = TRUE, scale = 0)


model.gam <- gam(medv ~
                   s(crim)+zn+s(indus)+chas+s(nox)+s(rm)+age
                 +s(dis)+rad+s(tax)+ptratio+s(black)+s(lstat),
                 data=train)
gam_summary=summary(model.gam)

AIC(model.gam)
BIC(model.gam)


Boston.gam.mse.train <-
  model.gam$dev/model.gam$df.res

#insample GAM
gam.pi.train=predict(model.gam)
mean((gam.pi.train - train$medv)^2)
mean(residuals(model.gam)^2)

#outsample GAM
gam.pi=predict(model.gam, test)
mean((gam.pi - test$medv)^2)

#Neural network####
install.packages("nnet")
library(nnet)
Boston.nnet <- nnet(medv ~ ., size = 4, data = train, linout = TRUE)

# Prediction using Neural Network####
#In-sample Prediction
boston.train.pred.nnet = predict(Boston.nnet,train)
mean((boston.train.pred.nnet - train$medv)^2)

# Out of sample Prediction
boston.test.pred.nnet = predict(Boston.nnet, test)
mean((boston.test.pred.nnet - test$medv)^2)
