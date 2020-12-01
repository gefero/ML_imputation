load("./data/data_EPH_2015_PROC2.RData")

df_train[df_train$p21 ==0, ]$p21<-100

library(caret)
library(tidyverse)
library(doParallel)

## Entrenamiento XGBoost
### Definición de grid de hiperparámetros
xgbGrid <- expand.grid(nrounds = c(100, 125, 130, 140, 150, 180, 200),                       
                       max_depth = c(5, 10, 15, 20),
                       colsample_bytree = seq(0.5, 0.9, 
                                              length.out = 5),
                       eta = c(0.3, 0.1, 0.05, 0.01),
                       gamma=c(0, 0.01),
                       min_child_weight = c(1, 0),
                       subsample = 1)



cl <- makeCluster(8, type='PSOCK', outfile='')
registerDoParallel(cl)

t0<-proc.time()
xgb_model<-train(p21 ~ ., data = df_train, 
                 method = "xgbTree", 
                 trControl = fitControl, 
                 verbose = TRUE, 
                 tuneGrid = xgbGrid,
                 metric='RMSE')
proc.time() - t0
stopCluster(cl)

saveRDS(xgb_model, file='./models/20200904_xgb_model_sin_ceros.rds', compress=FALSE)

#    nrounds max_depth eta gamma colsample_bytree min_child_weight subsample
#     200         5 0.1  0.01              0.6                0         1



xgb_model<-readRDS('./models/20200904_xgb_model_sin_ceros.rds')
cl <- makeCluster(8, type='PSOCK', outfile='')
registerDoParallel(cl)

xgb_final<-train(p21 ~ ., data = df_train, 
                 method = "xgbTree", 
                 trControl = fitControl2, 
                 verbose = FALSE, 
                 tuneGrid = xgb_model$bestTune,
                 metric='RMSE')
stopCluster(cl)
saveRDS(xgb_final, "./models/xgb_final_eval_sin_ceros.rds", compress=FALSE)

xbg<-readRDS("./models/20200928_xgb_final_f_sin_ceros.rds")

# cl <- makeCluster(8, type='PSOCK', outfile='')
# registerDoParallel(cl)
#                                         
# xgb_final_f<-train(p21 ~ ., data = df_train, 
#                  method = "xgbTree",
#                  tuneGrid = xgb_model$bestTune)
# stopCluster(cl)





## GRID FINAL

xgbGrid <- expand.grid(nrounds = 200,                       
                       max_depth = 5,
                       colsample_bytree = 0.6,
                       eta = 0.1,
                       gamma=0.01,
                       min_child_weight = 0,
                       subsample = 1)


        
cl <- makeCluster(8, type='PSOCK', outfile='')
registerDoParallel(cl)
        
xgb_final_f<-train(p21 ~ ., 
                   data = df_train, 
                   method = "xgbTree",
                   trControl = fitControl2, 
                   verbose = FALSE, 
                   metric='RMSE',
                   tuneGrid = xgbGrid)
                
stopCluster(cl)
saveRDS(xgb_final_f, "./models/20200928_xgb_final_f_sin_ceros.rds")



                
                #RMSE      Rsquared   MAE     
#3291.117  0.6754985  2017.501          

#Tuning parameter 'nrounds' was held constant at a value of 200
#Tuning parameter
#parameter 'min_child_weight' was held constant at a value of 0
#Tuning parameter
#'subsample' was held constant at a value of 1
        