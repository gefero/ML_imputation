# Evaluación pobreza
library(eph)

df<-get_microdata(year=2018, trimester=2, type='individual')

sum(df$PONDERA)
as.numeric(calculate_tabulates(df, 'P47T', weights='PONDERA')[1,])[2] +
as.numeric(calculate_tabulates(df, 'T_VI', weights='PONDERA')[1,])[2]

#2016: 15.1%
#2017: 14.2%
#2018: 14.2%
imp<-c(0.084202575288903,
   0.0685485610411,
   0.067979530947916,
   0.07543782737441,
   0.093440493723079,
   0.103284211011861,
   0.113294710541853,
   0.138604302045211,
   0.149407380980382,
   0.149812338592716,
   0.132179532905799,
   0.149805567543764,
   0.150772203540126,
   0.142681326351032,
   0.14109070422803
)*100

imp <- data.frame(year=2004:2018, imput=imp)

library(ggplot2)

ggplot(data=imp, aes(x=year, y=imput)) +
        geom_line(col='red') + 
        geom_point() +
        geom_vline(xintercept = 2007, color='blue', linetype='dashed') +
        geom_vline(xintercept = 2016, color='blue', linetype='dashed') +
        labs(x='Año (II-Trimestre)',
             y='% casos S/R eningresos (individuos)') + 
   theme_minimal()

ggsave('./plots/01_NR_EPH.png')
