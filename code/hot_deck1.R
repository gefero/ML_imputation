load("./data/data_EPH_2015_PROC2.RData")

library(caret)
library(tidyverse)
library(doParallel)

library(hot.deck)

#write.table(df_train, './data/df_train.csv', sep=";", row.names = FALSE)

#rm(df_train)

#df_train<-read.csv('./data/df_train.csv', sep=";")

df_train[df_train$p21 ==0, ]$p21<-100

cv_error <- matrix(ncol=2, nrow=length(cv_index2))
colnames(cv_error)<-c('RMSE','MAE')

#f_tr <- data.matrix(df_train)
#df_tr <- as.matrix(df_tr)

for (i in 1:length(cv_index2)){
        cat("processing fold #", i, "\n")
        train_index <- cv_index2[[i]]
        df_tr <- df_train
        y_test <- df_tr[-train_index,]$p21
        df_tr[-train_index,]$p21 <- NA
        #cov <- df_tr %>% select(-p21) %>% names()
        
        y_pred<- hot.deck(df_tr, method='best.cell', m=1)$data[[1]]$p21
        y_pred<-y_pred[-train_index]
        
        #resid <- 10**y_pred - 10**y_test
        resid <- y_pred - y_test
        resid_sq <- resid^2
        resid_abs <- abs(resid)
        
        rmse <- sqrt(mean(resid_sq))
        mae <- mean(resid_abs)
        
        cv_error[i, 1]<-rmse
        cv_error[i, 2]<-mae
        cat("\t", "RMSE=", rmse, "\n")
        cat("\t", "MAE=", mae, "\n")
        
}

