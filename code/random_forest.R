
load("./data/data_EPH_2015_PROC2.RData")

library(caret)
library(tidyverse)
library(doParallel)

df_train[df_train$p21 ==0, ]$p21<-100

## Entrenamiento Random Forest
### Definición de grid de hiperparámetros

rfGrid <-  expand.grid(#mtry=1:(ncol(df_train)-1),
                       mtry=seq(1,25, 2),
                       splitrule='variance',
                       min.node.size=c(5, 10, 15, 20))


fitControl$allowParallel<-TRUE

### Activar multithread
cl <- makeCluster(6, type='PSOCK', outfile='')
registerDoParallel(cl)

### Entrenamiento - Tuning hiperparámetros
t0<-proc.time()
rf_model<- train(p21 ~ ., data = df_train, 
                 method = "ranger", 
                 trControl = fitControl, 
                 tuneGrid = rfGrid,
                 metric='RMSE')
proc.time() - t0
stopCluster(cl)

saveRDS(rf_model, "./models/20200925_rf_model_sin_ceros.rds")

#       Fitting mtry = 23, splitrule = variance, min.node.size = 10 on full training set

### Entrenamiento - Selección modelo final y train sobre todo el dataset
rf_model<-readRDS("./models/20200925_rf_model_sin_ceros.rds")

cl <- makeCluster(6, type='PSOCK', outfile='')
registerDoParallel(cl)

rf_final<-train(p21 ~ ., data = df_train, 
                method = "ranger", 
                trControl = fitControl2, 
                verbose = TRUE, 
                tuneGrid = rf_model$bestTune,
                metric='RMSE')

stopCluster(cl)
saveRDS(rf_final, "./models/20200925_rf_final_eval_sin_ceros.rds")
                
rf_final



rf_model_final<-readRDS("./models/20200925_rf_final_eval_sin_ceros.rds")


cl <- makeCluster(6, type='PSOCK', outfile='')
registerDoParallel(cl)
rf_final_f<-train(p21~., data=df_train,
                  method = "ranger",
                  tuneGrid = rf_final$bestTune)
stopCluster(cl)
        
saveRDS(rf_final_f, "./models/20200925_rf_final_f_sin_ceros.rds")


                


 ## Test en partition

tr_index <- createDataPartition(y=df_train$p21,
                                p=0.8,
                                list=FALSE)



rf_final_f<-train(p21~., data=df_train[tr_index,],
                  method = "ranger",
                  trControl=trainControl(method='none'),
                  tuneGrid = rf_model$bestTune)


y_rf <- predict(rf_final_f, df_train[-tr_index,])

sqrt(sum(df_train[-tr_index,]$p21 - y_rf)**2)
