german_credit = read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")


colnames(german_credit) = c("chk_acct", "duration", "credit_his", "purpose", 
                            "amount", "saving_acct", "present_emp", "installment_rate", "sex", "other_debtor", 
                            "present_resid", "property", "age", "other_install", "housing", "n_credits", 
                            "job", "n_people", "telephone", "foreign", "response")

View(german_credit)

# orginal response coding 1= good, 2 = bad we need 0 = good, 1 = bad
german_credit$response = german_credit$response - 1 # Run this once
str(german_credit)
german_credit$response = as.factor(german_credit$response)


present_resid <- as.factor(german_credit$present_resid)

set.seed(10743959)
subset = sample(nrow(german_credit), nrow(german_credit) * 0.75)
german_credit_train = german_credit[subset, ]
german_credit_test = german_credit[-subset, ]

dim(german_credit_train)
dim(german_credit_test)

# stepwise Selection Method
credit.glm0 <- glm(response ~ ., family = binomial, german_credit_train)
summary(credit.glm0)
AIC(credit.glm0)
BIC(credit.glm0)

#stepwise AIC
credit.glm.step <- step(credit.glm0)
summary(credit.glm.step)
AIC(credit.glm.step)
BIC(credit.glm.step)


#STEPWISE BIC
credit.glm.step2 <- step(credit.glm0, k = log(nrow(german_credit_train)))
summary(credit.glm.step2)
BIC(credit.glm.step2)
AIC(credit.glm.step2)

#### model choose to use
german_credit_model = glm(response ~ chk_acct + duration + purpose + credit_his + saving_acct + 
                            installment_rate + amount + sex + telephone + foreign + other_install + 
                            other_debtor + present_resid,  family = binomial(link = "logit"), data =  german_credit_train)
summary(german_credit_model)

#Deciding cut-off Probability
searchgrid = seq(0.01, 0.99, 0.01)
result = cbind(searchgrid, NA)
# assymmetric cost 
creditcost <- function(r, pi) {
  weight1 = 5
  weight0 = 1
  c1 = (r == 1) & (pi < pcut)  #logical vector - true if actual 1 but predict 0
  c0 = (r == 0) & (pi > pcut)  #logical vecotr - true if actual 0 but predict 1
  return(mean(weight1 * c1 + weight0 * c0))
}

for (i in 1:length(searchgrid)) {
  pcut <- result[i, 1]
  # assign the cost to the 2nd col
  result[i, 2] <- creditcost(german_credit_train$response, predict(german_credit_model, type = "response"))
}

plot(result, xlab="Cut Off Probabilities",ylab = "Cost in Training Set")
index.min <- which.min(result[, 2])  #find the index of minimum value
result[index.min, 2]  #min cost
result[index.min, 1] #cut off prob

# Predict on insample Training Data
german_credit.glm.prob.insample <- predict(german_credit_model, type = "response")
german_credit.glm.pred.insample  <- (german_credit.glm.prob.insample > (1/6) ) * 1
table(german_credit_train$response, german_credit.glm.pred.insample, dnn = c("Truth", "Predicted"))
mean(ifelse(german_credit_train$response != german_credit.glm.pred.insample, 1, 0))

install.packages("ROCR")
library(ROCR)
pred_train <- prediction(german_credit.glm.pred.insample, german_credit_train$response)
perf_Train <- performance(pred_train, "tpr", "fpr")
plot(perf_Train, colorize = TRUE)
as.numeric(performance(pred_train, 'auc')@y.values)


## Predict on outsample Testing Data
german_credit.glm.prob.outsample <- predict(german_credit_model, german_credit_test, type = "response")
german_credit.glm.pred.outsample  <- (german_credit.glm.prob.outsample > (1/6) ) * 1
table(german_credit_test$response, german_credit.glm.pred.outsample, dnn = c("Truth", "Predicted"))
#mis-classification rate
mean(ifelse(german_credit_test$response != german_credit.glm.pred.outsample, 1, 0))
# cost
creditcost(german_credit_test$response, german_credit.glm.pred.outsample)

pred_test <- prediction(german_credit.glm.pred.outsample, german_credit_test$response)
perf_test <- performance(pred_test, "tpr", "fpr")
plot(perf_test, colorize = TRUE)
as.numeric(performance(pred_test, 'auc')@y.values)

# Cross Validation
library(boot)
# whole data set
german_credit_cv <- glm(response ~ chk_acct + duration + purpose + credit_his + saving_acct + 
                          installment_rate + amount + sex + telephone + foreign + other_install + 
                          other_debtor + age, family = binomial, german_credit)
cv.result = cv.glm(german_credit, german_credit_cv, creditcost, 10)
cv.result$delta

# Classification Tree ####
library(rpart)
credit.rpart1 <- rpart(formula = response ~ ., data = german_credit_train, method = "class", 
                       parms = list(loss = matrix(c(0, 5, 1, 0), nrow = 2)))

credit.rpart <- prune(credit.rpart1, cp = 0.02)
plot(credit.rpart)
text(credit.rpart, cex=0.7) # text and plot need to be one after one another
text(credit.rpart)

install.packages("tree")
library(tree)
tree1 <- tree(response ~ ., data = german_credit_train)
m1 = prune.misclass(tree1, best = 17)
summary(m1)

install.packages("rattle")
install.packages("rpart.plot")
install.packages("RColorBrewer")
library(rattle)
library(rpart.plot)
library(RColorBrewer)
credit.rpart <- prune(credit.rpart1, cp = 0.02)
fancyRpartPlot(credit.rpart)

#InSample
credit.train.prob.tree1 = predict(credit.rpart, german_credit_train, type = "prob")
credit.train.pred.tree1 = predict(credit.rpart, german_credit_train, type = "class")
creditcost(german_credit_train$response, credit.train.prob.tree1)
mean(ifelse(german_credit_train$response != credit.train.pred.tree1, 1, 0))

pred_train_rpart <- prediction(credit.train.prob.tree1[, 2], german_credit_train$response)
perf_train_rpart <- performance(pred_train_rpart, "tpr", "fpr")
plot(perf_train_rpart, colorize = TRUE, main = "ROC Curve: Training Data")
as.numeric(performance(pred_train_rpart, 'auc')@y.values)

#OutSample
credit.test.prob.tree1 = predict(credit.rpart, german_credit_test, type = "prob")
credit.test.pred.tree1 = predict(credit.rpart, german_credit_test, type = "class")
creditcost(german_credit_test$response, credit.test.prob.tree1)
mean(ifelse(german_credit_test$response != credit.test.pred.tree1, 1, 0))

pred_test_rpart <- prediction(credit.test.prob.tree1[, 2], german_credit_test$response)
perf_test_rpart <- performance(pred_test_rpart, "tpr", "fpr")
plot(perf_test_rpart, colorize = TRUE, main = "ROC Curve: Testing Data")
as.numeric(performance(pred_test_rpart, 'auc')@y.values)


# 2. Generalized Additive Models (GAM) ####

library(mgcv)
credit.gam <- gam(response ~ chk_acct + s(duration) + credit_his + purpose + s(amount) + 
                    saving_acct + installment_rate + sex + other_debtor + present_resid + 
                    s(age) + other_install + housing + n_people + foreign, family = binomial, data = german_credit_train)
summary(credit.gam)
credit.gam$deviance/credit.gam$df.residual
AIC(credit.gam)
BIC(credit.gam)

plot(credit.gam, shade = TRUE, seWithMean = TRUE, scale = 0)


pcut.gam <- 1/6
prob.gam.in <- predict(credit.gam, german_credit_train, type = "response")
pred.gam.in <- (prob.gam.in >= pcut.gam) * 1
table(german_credit_train$response, pred.gam.in, dnn = c("Observation", "Prediction"))
mean(ifelse(german_credit_train$response != pred.gam.in, 1, 0))

pred_train_gam <- prediction(as.numeric(pred.gam.in), as.numeric(german_credit_train$response))
perf_train_gam <- performance(pred_train_gam, "tpr", "fpr")
plot(perf_train_gam, colorize = TRUE, main = "ROC Curve: Training Data")
as.numeric(performance(pred_train_gam, 'auc')@y.values)

# define the searc grid from 0.01 to 0.20
searchgrid = seq(0.01, 0.2, 0.01)
# result.gam is a 99x2 matrix, the 1st col stores the cut-off p, the 2nd
# column stores the cost
result.gam = cbind(searchgrid, NA)
# in the cost function, both r and pi are vectors, r=truth, pi=predicted
# probability
cost1 <- function(r, pi) {
  weight1 = 5
  weight0 = 1
  c1 = (r == 1) & (pi < pcut)  #logical vector - true if actual 1 but predict 0
  c0 = (r == 0) & (pi > pcut)  #logical vecotr - true if actual 0 but predict 1
  return(mean(weight1 * c1 + weight0 * c0))
}

for (i in 1:length(searchgrid)) {
  pcut <- result.gam[i, 1]
  # assign the cost to the 2nd col
  result.gam[i, 2] <- cost1(german_credit_train$response, predict(credit.gam, type = "response"))
}
plot(result.gam, ylab = "Cost in Training Set")

index.min <- which.min(result.gam[, 2])  #find the index of minimum value
result.gam[index.min, 2]  #min cost
result.gam[index.min, 1]

#out sample prediction
pcut <- result.gam[index.min, 1]
prob.gam.out <- predict(credit.gam, german_credit_test, type = "response")
pred.gam.out <- (prob.gam.out >= pcut) * 1
table(german_credit_test$response, pred.gam.out, dnn = c("Observation", "Prediction"))
mean(ifelse(german_credit_test$response != pred.gam.out, 1, 0))
creditcost(german_credit_test$response, pred.gam.out)

pred_test_gam <- prediction(as.numeric(pred.gam.out), as.numeric(german_credit_test$response))
perf_test_gam <- performance(pred_test_rpart, "tpr", "fpr")
plot(perf_test_gam, colorize = TRUE, main = "ROC Curve: Testing Data")
as.numeric(performance(pred_test_gam, 'auc')@y.values)

# 3. Linear Discriminant Analysis (LDA)####
library(MASS)
german_credit_train$response = as.factor(german_credit_train$response)
credit.lda <- lda(response ~ ., data = german_credit_train)
summary(credit.lda)

#In sample
prob.lda.in <- predict(credit.lda, data = credit.train)
pred.lda.in <- (prob.lda.in$posterior[, 2] >= 1/6) * 1
table(german_credit_train$response, pred.lda.in, dnn = c("Obs", "Pred"))
creditcost(german_credit_train$response, pred.lda.in)
mean(ifelse(german_credit_train$response != pred.lda.in, 1, 0))

pred_train_lda <- prediction(prob.lda.in$posterior[, 2], german_credit_train$response)
perf_train_lda <- performance(pred_train_lda, "tpr", "fpr")
plot(perf_train_lda, colorize = TRUE, main = "ROC Curve: Training Data")
as.numeric(performance(pred_train_lda, 'auc')@y.values)

#OutSample
prob.lda.out <- predict(credit.lda, newdata = german_credit_test)
pred.lda.out <- as.numeric((prob.lda.out$posterior[, 2] >= 1/6))
table(german_credit_test$response, pred.lda.out, dnn = c("Obs", "Pred"))
creditcost(german_credit_test$response, pred.lda.out)
mean(ifelse(german_credit_test$response != pred.lda.out, 1, 0))

pred_test_lda <- prediction(prob.lda.out$posterior[, 2], german_credit_test$response)
perf_test_lda <- performance(pred_test_lda, "tpr", "fpr")
plot(perf_test_lda, colorize = TRUE, main = "ROC Curve: Testing Data")
as.numeric(performance(pred_test_lda, 'auc')@y.values)

