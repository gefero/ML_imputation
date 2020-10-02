load("./data/data_EPH_2015_II.RData")

rm(data_tr)
rm(data_te)

library(caret)
library(tidyverse)
library(doParallel)
library(glmnet)

eval <- function(y, ypred){
        precision <- posPredValue(ypred, y, positive="1")
        
        recall <- sensitivity(ypred, y, positive="1")
        
        f1 <- (2 * precision * recall) / (precision + recall)
        
        return(tibble(precision=precision,
                      recall=recall,
                      f1=f1)
        )
}

data$pp03i<-factor(data$pp03i, labels=c('1-SI', '2-No', '9-NS'))

data$intensi<-factor(data$intensi, labels=c('1-Sub_dem', '2-SO_no_dem', 
                                            '3-Ocup.pleno', '4-Sobreoc',
                                            '5-No trabajo', '9-NS'))

data$pp07a<-factor(data$pp07a, labels=c('0-NC',
                                        '1-Menos de un mes',
                                        '2-1 a 3 meses',
                                        '3-3 a 6 meses',
                                        '4-6 a 12 meses',
                                        '5-12 a 60 meses',
                                        '6-Más de 60 meses',
                                        '9-NS'))

df_train <- data
rm(data)

df_train_us <-  df_train %>%
        filter(imp_inglab1==0) %>%
        #mutate(aglomerado=fct_explicit_na(aglomerado)) %>%
        group_by(region) %>%
        sample_frac(0.26) %>%
        bind_rows(
                df_train %>%
                        filter(imp_inglab1==1)
        ) %>%
        ungroup() %>%
        sample_frac(1L)


X <- df_train_us %>%
        select(-imp_inglab1, -p21) %>%
        drop_na() %>%
        model.matrix(~.-1, .)


y <- df_train_us %>%
        drop_na %>%
        select(imp_inglab1) %>%
        pull()


k <- 5
set.seed(5198)
cv_index <- createFolds(y = df_train_us$imp_inglab1,
                        k=k,
                        list=FALSE,
                        returnTrain=TRUE)

cvfit <- cv.glmnet(X, y, intercept=FALSE, 
                   family='binomial', type.measure='class', 
                   nfolds = k, foldid = cv_index, 
                   standardize=TRUE)

plot(cvfit)
coef(cvfit, s = "lambda.1se")


eval(as.factor(predict(cvfit, newx = X, s = "lambda.1se", type='class')),
     as.factor(df_train_us$imp_inglab1))

X_new <- df_train %>%
        filter(imp_inglab1==0) %>%
        select(-imp_inglab1, -p21) %>%
        drop_na() %>%
        model.matrix(~.-1, .)


# Hacemos la predicción sobre los datos sin imputar del modelo, cambiando el threshold para obtener una proporción 
# similar a la de imp_inglab1n

df_mnar <- df_train %>%
        filter(imp_inglab1==0) %>%
        mutate(prob_nr = predict(cvfit, newx = X_new, s = "lambda.1se", type='response'),
               nr_estimate = case_when(
                                        prob_nr > 0.59 ~ 1,
                                        TRUE ~ 0)
               ) 

janitor::tabyl(df_mnar$nr_estimate) - janitor::tabyl(df_train$imp_inglab1)

rm(X, X_new, y, cvfit, cv_index, k)

save.image(file = "./data/mnar/mnar_data_EPH_2015_PROC2.RData")
