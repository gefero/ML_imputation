load("./data/mnar/mnar_data_EPH_2015_PROC2.RData")

df_train[df_train$p21 ==0, ]$p21<-100
df_mnar[df_mnar$p21 ==0, ]$p21<-100
df_train_us[df_train_us$p21 ==0, ]$p21<-100

library(caret)
library(tidyverse)
library(doParallel)
library(hot.deck)
library(keras)
        
### Definición de método de entrenamiento
set.seed(3369)
cv_index2 <- createFolds(y = df_mnar$p21,
                        k=5,
                        list=TRUE,
                        returnTrain=TRUE)


rf <- read_rds('./models/mnar/20200925_mnar_rf_model_sin_ceros.rds')
xgb <- read_rds('./models/mnar/20201123_mnar_xgb_model_sin_ceros.rds')
mlp <- load_model_hdf5('./models/mnar/20201129_mlp_final_sin_ceros.h5')

#f_tr <- data.matrix(df_train)
#df_tr <- as.matrix(df_tr)

calculate_error <- function(y_true, y_pred, type='RMSE'){
        resid <- y_true - y_pred
        
        if (type == 'RMSE'){
                resid_sq <- resid^2
                rmse <- sqrt(mean(resid_sq))
                return(rmse)
        }
        if (type == 'MAE'){
                resid_abs <- abs(resid)
                mae <- mean(resid_abs)
                return(mae)
        }
        
}


cv_error <- matrix(ncol=2, nrow=length(cv_index2))
model_array <- array(cv_error, dim=c(5,2,4), dimnames = list(1:5, c('RMSE','MAE'), 
                                                             c('hot_deck', 'rf', 'xgb', 'mlp')))
rm(cv_error)

for (i in 1:length(cv_index2)){
        cat("processing fold #", i, "\n")
        train_index <- cv_index2[[i]]
        df_te_ensamble <- df_mnar[-train_index,]
        y_te_ensamble <- df_mnar[-train_index,]$p21

        df_tr <- df_mnar
        y_test <- df_tr[-train_index,]$p21
        df_tr[-train_index,]$p21 <- NA
        
        ## HotDeck
        y_pred <- hot.deck(df_tr, method='best.cell', m=1)$data[[1]]$p21
        y_pred <- y_pred[-train_index]
        
        model_array[i, 1, 1] <- calculate_error(y_test, y_pred, type='RMSE')
        model_array[i, 2, 1]<- calculate_error(y_test, y_pred, type='MAE')
        
        ## RF
        y_pred_rf <- predict(rf, df_te_ensamble)
        model_array[i, 1, 2] <- calculate_error(y_te_ensamble, y_pred_rf, type='RMSE')
        model_array[i, 2, 2] <- calculate_error(y_te_ensamble, y_pred_rf, type='MAE')

        ## XGB
        y_pred_xgb <- predict(xgb, df_te_ensamble)

        model_array[i, 1, 3] <- calculate_error(y_te_ensamble, y_pred_xgb, type='RMSE')
        model_array[i, 2, 3] <- calculate_error(y_te_ensamble, y_pred_xgb, type='MAE')
        
        ## MLP
        
        X <- df_te_ensamble %>%
                select(-p21)
       
        dmy <- dummyVars(~., data=X)
        
        X <- as.matrix(predict(dmy, newdata = X))
        
        
        y_pred_mlp <- predict(mlp, X)
        model_array[i, 1, 4] <- calculate_error(y_te_ensamble, y_pred_mlp, type='RMSE') 
        model_array[i, 2, 4] <- calculate_error(y_te_ensamble, y_pred_mlp, type='MAE')

}

apply(model_array, c(2,3), mean)
