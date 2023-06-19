# Preparación de conjuntos de datos
# Winsonsin Diagnosis y Coimbra

# Instalar librerías
install.packages("corrplot")
install.packages("caret")
install.packages("dplyr")
install.packages("xtable")

# Iniciar la medición del tiempo de ejecución
hora.inicio <- Sys.time()

# Llamar librerías que se van a utilizar
library("corrplot")
library(caret)
library("dplyr")
library(xtable)

# 1) Establecer directorio de trabajo
setwd("H:/Mi unidad/Maestría/TFM/7. Experimentación")

# 2) Cargado de los conjuntos de datos

# 2.1) Conjunto de datos Wisconsin - Diagnóstico

# a) Leer datos del archivo seleccionado
data_orig_wdbc <- read.table("Datasets/wdbc.data", fileEncoding="UTF-8", sep=",")

# b) Visualizar información descriptiva del conjunto de datos
str(data_orig_wdbc)

# c) Visualizar información estadística acerca del conjunto de datos
summary(data_orig_wdbc)

# d) Eliminación de la primera variable relacionada con el número de identificación del caso
data_mod_wdbc <- data.frame(data_orig_wdbc[,-1])

# e) Renombrar las columnas para una mejor identificación de las variables
names(data_mod_wdbc) <- c('diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
                          'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
                          'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                          'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                          'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
                          'fractal_dimension_se', 'radius_worst', 'texture_worst',
                          'perimeter_worst', 'area_worst', 'smoothness_worst',
                          'compactness_worst', 'concavity_worst', 'concave_points_worst',
                          'symmetry_worst', 'fractal_dimension_worst')


summary(data_mod_wdbc)

# f) Verificar existencia de valores pérdidos mediante el conteo de los mismos
sum(is.na(data_mod_wdbc))
##No existen valores faltantes en el conjunto de datos

# Función para obtener valores pérdidos del conjunto de datos 
getValuesNA <- function(df){
  m<-c()
  for(i in colnames(df)){
    x<-sum(is.na(df[,i]))
    m<-append(m,x)
    m<-append(m,nrow(df)-x) 
  }
  
  a<-matrix(m,nrow=2)
  rownames(a)<-c("TRUE","FALSE")
  colnames(a)<-colnames(df)
  
  return(a)
}

dataframeNA = getValuesNA(data_mod_wdbc)

# Graficar datos de valores pérdidos
par(mar=c(3, 12, 3, 3))
barplot(dataframeNA,
        main = "Valores pérdidos del conjunto de datos",xlab = "Frecuencia",
        col = c("#4dffd2","#ff9999"),beside=TRUE,
        horiz = TRUE,
        las=1,
        xlim=c(0,700))
legend("topright", cex=0.6, pch = 15,
       c("Valores NA","Valores normales"),
       fill = c("#4dffd2","#ff9999"))


# I. TODAS LAS VARIABLES SIN NORMALIZAR

# Inicializar semilla
set.seed(9080100)

# Convertir variable categórica a factor
data_mod_wdbc$diagnosis <- as.factor(data_mod_wdbc$diagnosis)

# Dividir el conjunto de datos en datos de entrenamiento y datos de prueba
train.index.comp <- data_mod_wdbc$diagnosis %>% createDataPartition(p = 0.75, list = FALSE)
train.comp.wdbc  <- data_mod_wdbc[train.index.comp,]
test.comp.wdbc <- data_mod_wdbc[-train.index.comp,]

# Guardar datos en csv y latex
write.csv(data_mod_wdbc, "Datasets/Final/csv/data_mod_wdbc.csv", row.names=TRUE)
write.csv(train.comp.wdbc, "Datasets/Final/csv/train.comp.wdbc.csv", row.names=TRUE)
write.csv(test.comp.wdbc, "Datasets/Final/csv/test.comp.wdbc.csv", row.names=TRUE)

print(xtable(data_mod_wdbc, type = "latex"), file = "Datasets/Final/latex/data_mod_wdbc.tex")
print(xtable(train.comp.wdbc, type = "latex"), file = "Datasets/Final/latex/train.comp.wdbc.tex")
print(xtable(test.comp.wdbc, type = "latex"), file = "Datasets/Final/latex/test.comp.wdbc.tex")

# II. TODAS LAS VARIABLES CON NORMALIZACIÓN

# Escalar variables
train.comp_N.wdbc  <- as.data.frame(scale(data_mod_wdbc[train.index.comp,-1]))
test.comp_N.wdbc <- as.data.frame(scale(data_mod_wdbc[-train.index.comp,-1]))

train.comp_N.wdbc['diagnosis'] <- data_mod_wdbc[train.index.comp,1]
test.comp_N.wdbc['diagnosis'] <- data_mod_wdbc[-train.index.comp,1]

# Guardar datos en csv y latex
write.csv(train.comp_N.wdbc, "Datasets/Final/csv/train.comp_N.wdbc.csv", row.names=TRUE)
write.csv(test.comp_N.wdbc, "Datasets/Final/csv/test.comp_N.wdbc.csv", row.names=TRUE)

print(xtable(train.comp_N.wdbc, type = "latex"), file = "Datasets/Final/latex/train.comp_N.wdbc.tex")
print(xtable(test.comp_N.wdbc, type = "latex"), file = "Datasets/Final/latex/test.comp_N.wdbc.tex")

# III. VARIABLES SIN COLINEALIDAD Y SIN NORMALIZAR

# Obtener matriz de correlación entre las variables
matrix_corr <- cor(data_mod_wdbc[,-1])

# Encontrar variables correlacionadas y evitar la colinealidad
data.opt_SE.wdbc <- data.frame(data_mod_wdbc$diagnosis, data_mod_wdbc[,-findCorrelation(matrix_corr, cutoff = 0.75)])

#Renombrar primera columna
colnames(data.opt_SE.wdbc)[1] <- "diagnosis"

# Visualizar número de variables resultante del proceso anterior
ncol(data.opt_SE.wdbc)

## 14 variables

# Dividir el conjunto de datos en datos de entrenamiento y datos de prueba
train.op_SE.wdbc  <- data.opt_SE.wdbc[train.index.comp,]
test.op_SE.wdbc <- data.opt_SE.wdbc[-train.index.comp,]

# Guardar datos en csv y latex
write.csv(train.op_SE.wdbc, "Datasets/Final/csv/train.op_SE.wdbc.csv", row.names=TRUE)
write.csv(test.op_SE.wdbc, "Datasets/Final/csv/test.op_SE.wdbc.csv", row.names=TRUE)

print(xtable(train.op_SE.wdbc, type = "latex"), file = "Datasets/Final/latex/train.op_SE.wdbc.tex")
print(xtable(test.op_SE.wdbc, type = "latex"), file = "Datasets/Final/latex/test.op_SE.wdbc.tex")

# IV. VARIABLES SIN COLINEALIDAD Y CON NORMALIZACIÓN

# Escalar variables
train.op_SE_N.wdbc  <- as.data.frame(scale(data.opt_SE.wdbc[train.index.comp,-1]))
test.op_SE_N.wdbc <- as.data.frame(scale(data.opt_SE.wdbc[-train.index.comp,-1]))

train.op_SE_N.wdbc['diagnosis'] <- data.opt_SE.wdbc[train.index.comp,1]
test.op_SE_N.wdbc['diagnosis'] <- data.opt_SE.wdbc[-train.index.comp,1]

# Guardar datos en csv y latex
write.csv(train.op_SE_N.wdbc, "Datasets/Final/csv/train.op_SE_N.wdbc.csv", row.names=TRUE)
write.csv(test.op_SE_N.wdbc, "Datasets/Final/csv/test.op_SE_N.wdbc.csv", row.names=TRUE)

print(xtable(train.op_SE_N.wdbc, type = "latex"), file = "Datasets/Final/latex/train.op_SE_N.wdbc.tex")
print(xtable(test.op_SE_N.wdbc, type = "latex"), file = "Datasets/Final/latex/test.op_SE_N.wdbc.tex")

# 2.1) Conjunto de datos - Breast Cancer Coimbra Data Set

# a) Leer datos del archivo seleccionado
data_orig_coimbra <- read.table("Datasets/dataR2.csv", fileEncoding="UTF-8", sep=",", header = TRUE)

# b) Visualizar información descriptiva del conjunto de datos
str(data_orig_coimbra)

# c) Renombrar las columnas para una mejor identificación de las variables
names(data_orig_coimbra) <- c('age', 'bmi', 'glucose', 'insulin',
                              'homa', 'leptin', 'adiponectin', 'resistin',
                              'mcp_1', 'result_bc')

# d) Visualizar información estadística acerca del conjunto de datos
summary(data_orig_coimbra[,-10])

# e) Verificar existencia de valores pérdidos mediante el conteo de los mismos
sum(is.na(data_orig_coimbra))
##No existen valores faltantes en el conjunto de datos

# I. TODAS LAS VARIABLES SIN NORMALIZAR

# Inicializar semilla
set.seed(212135)

# Convertir variable categórica a factor
data_orig_coimbra$result_bc <- as.factor(data_orig_coimbra$result_bc)

# Modificar los valores de los niveles de la variable result_bc
levels(data_orig_coimbra$result_bc)=c("S","P")

# Dividir el conjunto de datos en datos de entrenamiento y datos de prueba
train.index.comp.coimbra <- data_orig_coimbra$result_bc %>% createDataPartition(p = 0.75, list = FALSE)
train.comp.coimbra  <- data_orig_coimbra[train.index.comp.coimbra,]
test.comp.coimbra <- data_orig_coimbra[-train.index.comp.coimbra,]

# Guardar datos en csv y latex
write.csv(data_orig_coimbra, "Datasets/Final/csv/data_orig_coimbra.csv", row.names=TRUE)
write.csv(train.comp.coimbra, "Datasets/Final/csv/train.comp.coimbra.csv", row.names=TRUE)
write.csv(test.comp.coimbra, "Datasets/Final/csv/test.comp.coimbra.csv", row.names=TRUE)

print(xtable(data_orig_coimbra, type = "latex"), file = "Datasets/Final/latex/data_orig_coimbra.tex")
print(xtable(train.comp.coimbra, type = "latex"), file = "Datasets/Final/latex/train.comp.coimbra.tex")
print(xtable(test.comp.coimbra, type = "latex"), file = "Datasets/Final/latex/test.comp.coimbra.tex")

# II. TODAS LAS VARIABLES CON NORMALIZACIÓN

# Escalar variables
train.comp_N.coimbra  <- as.data.frame(scale(data_orig_coimbra[train.index.comp.coimbra,-10]))
test.comp_N.coimbra <- as.data.frame(scale(data_orig_coimbra[-train.index.comp.coimbra,-10]))

train.comp_N.coimbra['result_bc'] <- data_orig_coimbra[train.index.comp.coimbra,10]
test.comp_N.coimbra['result_bc'] <- data_orig_coimbra[-train.index.comp.coimbra,10]

# Guardar datos en csv y latex
write.csv(train.comp_N.coimbra, "Datasets/Final/csv/train.comp_N.coimbraa.csv", row.names=TRUE)
write.csv(test.comp_N.coimbra, "Datasets/Final/csv/test.comp_N.coimbra.csv", row.names=TRUE)

print(xtable(train.comp_N.coimbra, type = "latex"), file = "Datasets/Final/latex/train.comp_N.coimbra.tex")
print(xtable(test.comp_N.coimbra, type = "latex"), file = "Datasets/Final/latex/test.comp_N.coimbra.tex")

# III. VARIABLES SIN COLINEALIDAD Y SIN NORMALIZAR

# Crear una variable con los datos del dataset original
data_mod_coimbra <- data_orig_coimbra[,-10]

# Obtener matriz de correlación entre las variables
matrix_corr <- cor(data_mod_coimbra)

# Encontrar variables correlacionadas y evitar la colinealidad
data.opt_SE.coimbra <- data.frame(data_orig_coimbra$result_bc, data_mod_coimbra[,-findCorrelation(matrix_corr, cutoff = 0.75)])

#Renombrar primera columna
colnames(data.opt_SE.coimbra)[1] <- "result_bc"

# Visualizar número de variables resultante del proceso anterior
ncol(data.opt_SE.coimbra)

## 9 variables

# Dividir el conjunto de datos en datos de entrenamiento y datos de prueba
train.op_SE.coimbra  <- data.opt_SE.coimbra[train.index.comp.coimbra,]
test.op_SE.coimbra <- data.opt_SE.coimbra[-train.index.comp.coimbra,]

# Guardar datos en csv y latex
write.csv(train.op_SE.coimbra, "Datasets/Final/csv/train.op_SE.coimbra.csv", row.names=TRUE)
write.csv(test.op_SE.coimbra, "Datasets/Final/csv/test.op_SE.coimbra.csv", row.names=TRUE)

print(xtable(train.op_SE.coimbra, type = "latex"), file = "Datasets/Final/latex/train.op_SE.coimbra.tex")
print(xtable(test.op_SE.coimbra, type = "latex"), file = "Datasets/Final/latex/test.op_SE.coimbra.tex")

# IV. VARIABLES SIN COLINEALIDAD Y CON NORMALIZACIÓN

# Escalar variables
train.op_SE_N.coimbra  <- as.data.frame(scale(data.opt_SE.coimbra[train.index.comp.coimbra,-1]))
test.op_SE_N.coimbra <- as.data.frame(scale(data.opt_SE.coimbra[-train.index.comp.coimbra,-1]))

train.op_SE_N.coimbra['result_bc'] <- data.opt_SE.coimbra[train.index.comp.coimbra,1]
test.op_SE_N.coimbra['result_bc'] <- data.opt_SE.coimbra[-train.index.comp.coimbra,1]

# Guardar datos en csv y latex
write.csv(train.op_SE_N.coimbra, "Datasets/Final/csv/train.op_SE_N.coimbra.csv", row.names=TRUE)
write.csv(test.op_SE_N.coimbra, "Datasets/Final/csv/test.op_SE_N.coimbra.csv", row.names=TRUE)

print(xtable(train.op_SE_N.coimbra, type = "latex"), file = "Datasets/Final/latex/train.op_SE_N.coimbra.tex")
print(xtable(test.op_SE_N.coimbra, type = "latex"), file = "Datasets/Final/latex/test.op_SE_N.coimbra.tex")

# Finalizar la medición del tiempo de ejecución
hora.final <- Sys.time()

# Obtener tiempo de ejecución en horas
tiempo.ejec <- (hora.final - hora.inicio)[[1]]/3600

# Guardar valor del tiempo de ejecución
file.connection <- file("Datasets/Final/execution_time_read_and_write_dataset.txt")
writeLines(c("Tiempo de ejecución",paste(tiempo.ejec,"horas",sep=" ")), file.connection)
close(file.connection)










