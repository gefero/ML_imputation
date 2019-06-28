load("./data/data_EPH_2015_PROC2.RData")

library(caret)
library(tidyverse)
library(doParallel)
library(keras)

df_train[df_train$p21 ==0, ]$p21<-100


X_imp <- df_imp %>%
        select(-p21)

dmy <- dummyVars(~., data=X_imp)

X_imp <- as.matrix(predict(dmy, newdata = X_imp))




rf_final<-readRDS('./models/rf_final_f_sin_ceros.rds')
xgb_final<-readRDS('./models/xgb_final_f_sin_ceros.rds')
mlp_final<-load_model_hdf5('./models/mlp_final_f_sin_ceros.h5')

# PREDICCION SOBRE DATOS IMPUTADOS (df_imp)

y_preds_rf <- predict(rf_final, df_imp)
y_preds_xgb <- predict(xgb_final, df_imp)
y_preds_mlp <- as.vector(predict(mlp_final, X_imp))
y_preds_hd <- df_imp$p21


preds <- as.data.frame(cbind(y_preds_rf, 
                             y_preds_xgb, 
                             y_preds_mlp, 
                             y_preds_hd))
colnames(preds)<-c('RandomForest', 
                   "XGBoost", 
                   "MLPerceptron", 
                   "HotDeck")


preds[preds$XGBoost < 0,]$XGBoost<-min(preds[preds$XGBoost > 0,]$XGBoost)
preds[preds$MLPerceptron < 0,]$MLPerceptron<-min(preds[preds$MLPerceptron > 0,]$MLPerceptron)
preds_tidy <- gather(preds)
colnames(preds_tidy)<-c('method', 'value')

# COMPARACION DE DISTRIBUCIONES DE DATOS IMPUTADOS

ggplot(data=preds_tidy) + 
        geom_density(aes(x=value, fill=method), alpha=0.4) +
        theme_minimal()

ggsave('./plots/G3_density_imp.png')

ggplot(data=preds_tidy) +
        geom_boxplot(aes(x=method, y=value, fill=method)) + 
        theme_minimal()

ggsave('./plots/G4_boxplot_imp.png')


preds_tidy %>% 
        group_by(method) %>%
        summarize(Min=min(value),
                  Q1=quantile(value, probs=0.25),
                  Mediana=quantile(value, probs=0.5),
                  Media=mean(value),
                  Q3=quantile(value, probs=0.75),
                  Max=max(value),
                  Std.Dev=sd(value),
                  CV=Std.Dev/Media,
                  MAD=mad(value))


# COMPARACION DE TOTAL DE CASOS 

df_train<-df_train %>%
        bind_rows(df_imp %>%
                bind_cols(preds))


df_train[is.na(df_train$RandomForest),]$RandomForest<-df_train[is.na(df_train$RandomForest),]$p21
df_train[is.na(df_train$XGBoost),]$XGBoost<-df_train[is.na(df_train$XGBoost),]$p21
df_train[is.na(df_train$MLPerceptron),]$MLPerceptron<-df_train[is.na(df_train$MLPerceptron),]$p21
df_train[is.na(df_train$HotDeck),]$HotDeck<-df_train[is.na(df_train$HotDeck),]$p21


df_train %>%
        select(cat_ocup, RandomForest, XGBoost, MLPerceptron, HotDeck) %>%
        gather(method, value, RandomForest:HotDeck) %>%
        #rename(method=key) %>%
        ggplot() + 
        geom_density(aes(x=value, fill=method), alpha=0.4) +
        #facet_grid(~cat_ocup) +
        theme_minimal()

ggsave('./plots/G4_density_comp.png')


df_train %>%
        filter(cat_ocup!='Trabajador familiar sin remuneraciÃ³n') %>%
        select(cat_ocup, RandomForest, XGBoost, MLPerceptron, HotDeck) %>%
        gather(method, value, RandomForest:HotDeck) %>%
        #rename(method=key) %>%
        ggplot() + 
                geom_density(aes(x=value, fill=method), alpha=0.4) +
                facet_grid(~cat_ocup) +
                theme_minimal()

ggsave('./plots/G5_density_comp_cat_ocup.png')

df_train %>%
        select(RandomForest, XGBoost, MLPerceptron, HotDeck) %>%
        gather() %>%
        rename(method=key) %>%
        group_by(method) %>%
        summarize(Min=min(value),
                  Q1=quantile(value, probs=0.25),
                  Mediana=quantile(value, probs=0.5),
                  Media=mean(value),
                  Q3=quantile(value, probs=0.75),
                  Max=max(value),
                  Std.Dev=sd(value),
                  MAD=mad(value))
