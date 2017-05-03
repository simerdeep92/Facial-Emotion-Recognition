library(mlbench)
library(caret)
dataset <- read.csv("landmark_data_relative.csv",header = FALSE)
initial_dataset <- dataset[,c(2:ncol(dataset))]

#write.csv(dataset,'landmarks_reduced_relative.csv',row.names = FALSE)

X_data <- initial_dataset [c(2:ncol(initial_dataset))]
pca.data = prcomp(X_data)
plot(pca.data,type='l')
#Y <- dataset[,2]

dataset = pca.data$x[,1:5]
dataset = cbind.data.frame(dataset,initial_dataset[,1])
colnames(dataset) = c('c1','c2','c3','c4','c5','V5')
# 10 fold cross validation
control <- trainControl(method="repeatedcv", number=10, repeats=3)
#random number seed
seed <- 7
#Evaluation accuracy
metric <- "Accuracy"

## different machine learning algorithm
# Linear Discriminant Analysis
set.seed(seed)
fit.lda <- train(V5~., data=dataset, method="lda", metric=metric, preProc=c("center", "scale"), trControl=control)
# Logistic Regression
set.seed(seed)
#fit.glm <- train(V2~., data=dataset, method="glm", metric=metric, trControl=control)
# GLMNET
set.seed(seed)
#fit.glmnet <- train(V2~., data=dataset, method="glmnet", metric=metric, preProc=c("center", "scale"), trControl=control)
# SVM Radial
set.seed(seed)
fit.svmRadial <- train(V5~., data=dataset, method="svmRadial", metric=metric, preProc=c("center", "scale"), trControl=control, fit=FALSE)
# kNN
set.seed(seed)
fit.knn <- train(V5~., data=dataset, method="knn", metric=metric, preProc=c("center", "scale"), trControl=control)
# Naive Bayes
set.seed(seed)
fit.nb <- train(V5~., data=dataset, method="nb", metric=metric, trControl=control)
# CART
set.seed(seed)
fit.cart <- train(V5~., data=dataset, method="rpart", metric=metric, trControl=control)
# C5.0
set.seed(seed)
#fit.c50 <- train(V2~., data=dataset, method="C5.0", metric=metric, trControl=control)
# Bagged CART
set.seed(seed)
fit.treebag <- train(V5~., data=dataset, method="treebag", metric=metric, trControl=control)
# Random Forest
set.seed(seed)
fit.rf <- train(V5~., data=dataset, method="rf", metric=metric, trControl=control)
# Stochastic Gradient Boosting (Generalized Boosted Modeling)
set.seed(seed)
fit.gbm <- train(V5~., data=dataset, method="gbm", metric=metric, trControl=control, verbose=FALSE)

results <- resamples(list(lda=fit.lda, 
                          svm=fit.svmRadial, knn=fit.knn, nb=fit.nb, cart=fit.cart, 
                          bagging=fit.treebag, rf=fit.rf, gbm=fit.gbm))
# Table comparison
summary(results)