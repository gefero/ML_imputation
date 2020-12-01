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
library(keras)

set.seed(77887)
cv_index <- createFolds(y = df_mnar$p21,
                        k=5,
                        list=TRUE,
                        returnTrain=TRUE)

X <- df_mnar %>%
        select(-p21)

y <- df_mnar$p21
#y_log <- log10(df_train$p21)

dmy <- dummyVars(~., data=X)

X <- as.matrix(predict(dmy, newdata = X))


# ideas thermo encoding para variables categóricas
# replicar el pipeline del paper


# Función para generarl MLP
build_sequential <- function(n_hidden=3, 
                             n_units=128*2, 
                             act_function='tanh',
                             in_shape=dim(X)[2],
                             drop_rate=0.5,
                             loss_function="mae",
                             metr='mse',
                             optm=optimizer_rmsprop()){
        
        #on.exit(keras::backend()$clear_session())
        
        if (is.numeric(n_units)){
                n_units <- rep(n_units, n_hidden)
        }
        
        
        model <- keras_model_sequential()
        
        for (i in 1:n_hidden){
                model %>%
                        layer_dense(units = n_units[i], activation = act_function, input_shape = c(in_shape)) %>% 
                        layer_dropout(rate=drop_rate)
        }
                
        model %>% layer_dense(units = 1)
        
        model %>% compile(
                loss = loss_function,
                optimizer = optm,
                metrics = list(metr)
        )
         
        return(model)
}



#K <- backend()
#metric_rmse <- function(y_true, y_pred) {
#        K$sqrt(metric_mean_squared_error(y_true, y_pred))
#}





#n_h <- 4
#n_u <- 128*3
#loss<-'mse'


cross_val_mlp<-function(n_h, n_u, index=cv_index, loss, X=X, dep=y_log){
        cv_error <- matrix(ncol=2, nrow=length(cv_index))
        colnames(cv_error)<-c('RMSE','MAE')
        

        for (i in 1:length(index)){
                cat("processing fold #", i, "\n")
                train_index <- index[[i]]
                X_test <- X[-train_index,]
                y_test<-dep[-train_index]
                #y_test <- y_log[-train_index]
                #y_test <- y[-train_index]
                
                X_train <- X[train_index,]
                y_train <- dep[train_index]
                #y_train <- y_log[train_index]
                #y_train <- y[train_index]
                
                model <- build_sequential(n_hidden = n_h,
                                          n_units = n_u,
                                          #n_units = c(384, 256, 128)
                                          loss_function=loss)
                
                #early_stop <- callback_early_stopping(monitor = "val_loss", 
                #                                      patience = 20)
                
                model %>% fit(
                        X_train,
                        y_train,
                        epochs=400,
                #        callbacks=list(early_stop),
                        verbose=0
                )
                
                #results <- model %>% evaluate(X_test, y_test)
                #mae<-c(mae, results$mean_absolute_error)
                #mse<-c(mse, results$mean_squared_error)
                #rmse<-c(rmse, sqrt(mse))
                y_pred <- predict(model, X_test)
                #resid <- 10**y_pred - 10**y_test
                resid <- y_pred - y_test
                resid_sq <- resid**2
                resid_abs <- abs(resid)
                rmse <- sqrt(mean(resid_sq))
                mae <- mean(resid_abs)
                
                cv_error[i, 1]<-rmse
                cv_error[i, 2]<-mae
                cat("\t", "RMSE=", rmse, "\n")
                cat("\t", "MAE=", mae, "\n")
        }
        return(list(model=model, errors=cv_error, 
                    cv_error=apply(cv_error, 2, mean)))
        
}

t0<-proc.time()
mlp_eval<- cross_val_mlp(n_h=3, 
                         n_u=512, 
                         index=cv_index, 
                         loss='mse', 
                         X=X, dep=y) # RMSE (cv_index)=4046
proc.time()-t0

mlp_eval$model %>% save_model_hdf5('./models/mnar/20201129_mlp_eval_final_sin_ceros.h5')
mlp_eval$cv_error

#RMSE      MAE 
#3970.614 2301.245 

#RMSE (cv_index2)=4200

# MODELO FINAL SOBRE TODO EL TRAINING SET
t0<-proc.time()

model_total <- build_sequential(n_hidden = 3, 
                                n_units = 512,
                                loss_function = 'mse')



model_total %>% fit(
                X,
                y,
                epochs=500,
                verbose=1
)
proc.time() - t0
model_total %>% save_model_hdf5('./models/mnar/20201129_mlp_final_sin_ceros.h5')

# GANADOR
#y, n_units3, nhidden=512, rmsprop, tanh, loss=mse CV error (RMSE)=4039
#model %>% save_model_hdf5('./models/logy_units3_hidden_512_tanh_rmsprop_mse.h5')


#log(y), n_units3, nhidden=384, rmsprop, tanh, loss=mse CV error (RMSE)=5800
#log(y), n_units3, nhidden=256, rmsprop, tanh, loss=mse CV error (RMSE)=4978.7
#log(y), n_units3, nhidden=128, rmsprop, tanh, loss=mse CV error (RMSE)=4588.7    





#y, n_units3, nhidden=128, rmsprop, tanh, loss=mae CV error (RMSE)=5564.9
#y, n_units3, nhidden=256, rmsprop, tanh, loss=mae CV error (RMSE)=4486.9
#y, n_units3, nhidden=384, rmsprop, tanh, loss=mae CV error (RMSE)=4373.5  
#y, n_units3, nhidden=512, rmsprop, tanh, loss=mae CV error (RMSE)=4274.8
#y, n_units4, nhidden=384, rmsprop, tanh, loss=mae CV error (RMSE)=4395.1





#log(y), n_units2, nhidden=384, rmsprop, tanh, loss=mse CV error (RMSE)=

#log(y), n_units2, nhidden=256, rmsprop, tanh, loss=mse CV error (RMSE)=
#log(y), n_units2, nhidden=128, rmsprop, tanh, loss=mse CV error (RMSE)=        

#y, n_units2, nhidden=384, rmsprop, tanh, loss=mae CV error (RMSE)=        
#y, n_units2, nhidden=256, rmsprop, tanh, loss=mae CV error (RMSE)=
#y, n_units2, nhidden=128, rmsprop, tanh, loss=mae CV error (RMSE)=

















param_search <- function(search='grid', n, x_train, y_train, param_grid){
        
        for (i in index_k){
                
                X_tr <- x_train[-i,]
                y_tr <- y_train[-i,]
                X_te <- x_train[i,]
                y_te <- y_train[i,]
                
                if (search=='random'){
                        
                        ix <- sample(1:nrow(param_grid), n)
                        param_grid <- param_grid[ix,]
                        
                        }
                
                for (p in 1:nrow(param_grid)){
                                
                        params <- param_grid[,p]
                                
                        model <- build_sequential(
                                n_hidden = params[1],
                                n_units = params[2],
                                act_function = params[3],
                                drop_rate = params[4],
                                optm = params[5]
                                        
                                )
                                
                        model %>% fit(
                                x_train, y_train, 
                                epochs = 200, 
                                #batch_size = 128, 
                                validation_split = 0.2,
                                callbacks = list(early_stop)
                        )
                                
                        }
                        
                }
                
        }
        




param_grid <- expand.grid(n_hidden=1:4,
                          n_units=c(32, 64, 128, 256),
                          act_function=c('tanh', 'relu'),
                          drop_rate=c(0.25, 0.5),
                          optm=list(optimizer_rmsprop(), optimizer_adam())
)
