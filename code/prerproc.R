
load("./data/data_EPH_2015_II.RData")

rm(data_tr)
rm(data_te)

library(caret)
library(tidyverse)
library(doParallel)


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


df_imp <- data %>%
        filter(imp_inglab1==1) %>%
        select(-imp_inglab1)

df <- data %>%
        filter(imp_inglab1==0) %>%
        select(-imp_inglab1)



df_train <- df
#df_train <- df[tr_index,]
#df_test <- df[-tr_index,]


### Indexex para cross validation
set.seed(9183)
cv_index <- createFolds(y = df_train$p21,
                        k=5,
                        list=TRUE,
                        returnTrain=TRUE)

fitControl <- trainControl(
        index=cv_index, 
        method="cv",
        number=5,
        verbose = TRUE,
        allowParallel=TRUE)


### Esquema de validación cruzada para estimación de error
set.seed(7412)
cv_index2 <- createFolds(y = df_train$p21,
                         k=5,
                         list=TRUE,
                         returnTrain=TRUE)

fitControl2 <- trainControl(
        index=cv_index2,
        method="cv",
        number=5,
        allowParallel=TRUE)

save.image(file = "./data/data_EPH_2015_PROC2.RData")
