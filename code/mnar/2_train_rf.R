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
library(doParallel)

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
        allowParallel=FALSE)

### Definición de grid de hiperparámetros

rfGrid <-  expand.grid(#mtry=1:(ncol(df_train)-1),
        mtry=seq(1,25, 2),
        splitrule='variance',
        min.node.size=c(5, 10, 15, 20))

fitControl$allowParallel<-TRUE

### Activar multithread
cl <- makeCluster(4, type='PSOCK', outfile='')
registerDoParallel(cl)

t0<-proc.time()
rf_model<- train(p21 ~ ., 
                 data = df_mnar, 
                 method = "ranger", 
                 trControl = fitControl, 
                 tuneGrid = rfGrid,
                 metric='RMSE')
proc.time() - t0
stopCluster(cl)

saveRDS(rf_model, "./models/mnar/20200925_mnar_rf_model_sin_ceros.rds")


