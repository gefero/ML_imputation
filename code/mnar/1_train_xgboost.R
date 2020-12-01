packages <- c("caret", "doParallel", "xgboost", "keras")
if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
        install.packages(setdiff(packages, rownames(installed.packages())))  
}

load("./data/mnar/mnar_data_EPH_2015_PROC2.RData")
rm(X, X_new, y, cvfit, cv_index, k)

df_train[df_train$p21 ==0, ]$p21<-100
df_mnar[df_mnar$p21 ==0, ]$p21<-100
df_train_us[df_train_us$p21 ==0, ]$p21<-100


library(caret)
library(tidyverse)

## Entrenamiento XGBoost
### Definición de método de entrenamiento
set.seed(77887)
cv_index <- createFolds(y = df_mnar$p21,
                        k=5,
                        list=TRUE,
                        returnTrain=TRUE)

fitControl <- trainControl(
        index=cv_index, 
        method="cv",
        number=5,
        verbose = TRUE,
        allowParallel=FALSE,
        search = "random")

### Definición de grid de hiperparámetros
xgbGrid <- expand.grid(nrounds = c(100, 125, 130, 140, 150, 180, 200),                       
                       max_depth = c(5, 10, 15, 20),
                       colsample_bytree = seq(0.5, 0.9, 
                                              length.out = 5),
                       eta = c(0.3, 0.1, 0.05, 0.01),
                       gamma=c(0, 0.01),
                       min_child_weight = c(1, 0),
                       subsample = 1)

t0<-proc.time()
xgb_model<-train(p21 ~ ., data = df_mnar, 
                 method = "xgbTree", 
                 trControl = fitControl, 
                 verbose = TRUE, 
                 tuneGrid = xgbGrid,
                 nthread = 1,
                 tuneLength = 300,
                 metric='RMSE')
proc.time() - t0

saveRDS(xgb_model, file='./models/mnar/20201123_mnar_xgb_model_sin_ceros.rds', compress=FALSE)

