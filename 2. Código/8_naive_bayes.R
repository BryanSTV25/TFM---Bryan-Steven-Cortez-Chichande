## Naive Bayes

# Instalar librerías
install.packages("ggplot2")
install.packages("extrafont")
install.packages("corrplot")
install.packages("caret")
install.packages("dplyr")
install.packages("naivebayes")
install.packages("xtable")

# Iniciar la medición del tiempo de ejecución
hora.inicio <- Sys.time()

# Llamar librerías que se van a utilizar
library("ggplot2")
library(extrafont)
library("corrplot")
library(caret)
library("dplyr")
library(naivebayes)
library(xtable)

# 0) Establecer directorio de trabajo
setwd("H:/Mi unidad/Maestría/TFM/7. Experimentación")

# 0.1) Descarga e importación de paquete de fuentes

# Listar fuentes disponibles
windowsFonts()

# Importar fuentes del paquete extrafont
font_import()

# Cargado de fuentes importadas
loadfonts(device = "win")

## CONJUNTO DE DATOS WISCONSIN - DIAGNÓSTICO

# 1) Lectura de las versiones del conjunto de datos Wisconsin - Diagnóstico

# 1.1) Leer datos de los archivos CSV

# Conjunto completo aplicado limpieza de datos
data_mod_wdbc <- read.table("Datasets/Final/csv/data_mod_wdbc.csv", fileEncoding="UTF-8", sep=",", header=TRUE)
data_mod_wdbc <- data_mod_wdbc[,-1]

# Conjuntos de entrenamiento y prueba conformados por todas las variables del conjunto de datos que se le aplicó la limpieza de datos
train.comp.wdbc <- read.table("Datasets/Final/csv/train.comp.wdbc.csv", fileEncoding="UTF-8", sep=",", header=TRUE)
train.comp.wdbc <- train.comp.wdbc[,-1]
test.comp.wdbc <- read.table("Datasets/Final/csv/test.comp.wdbc.csv", fileEncoding="UTF-8", sep=",", header=TRUE)
test.comp.wdbc <- test.comp.wdbc[,-1]

# Conjuntos de entrenamiento y prueba conformados por todas las variables escaladas del conjunto de datos que se le aplicó la limpieza de datos
train.comp_N.wdbc <- read.table("Datasets/Final/csv/train.comp_N.wdbc.csv", fileEncoding="UTF-8", sep=",", header=TRUE)
train.comp_N.wdbc <- train.comp_N.wdbc[,-1]
test.comp_N.wdbc <- read.table("Datasets/Final/csv/test.comp_N.wdbc.csv", fileEncoding="UTF-8", sep=",", header=TRUE)
test.comp_N.wdbc <- test.comp_N.wdbc[,-1]

# Conjuntos de entrenamiento y prueba conformados por las variables sin colinealidad y que no se les ha aplicado el escalado de valores 
train.op_SE.wdbc <- read.table("Datasets/Final/csv/train.op_SE.wdbc.csv", fileEncoding="UTF-8", sep=",", header=TRUE)
train.op_SE.wdbc <- train.op_SE.wdbc[,-1]
test.op_SE.wdbc <- read.table("Datasets/Final/csv/test.op_SE.wdbc.csv", fileEncoding="UTF-8", sep=",", header=TRUE)
test.op_SE.wdbc <- test.op_SE.wdbc[,-1]

# Conjuntos de entrenamiento y prueba conformados por las variables sin colinealidad y que se les ha aplicado el escalado de valores
train.op_SE_N.wdbc <- read.table("Datasets/Final/csv/train.op_SE_N.wdbc.csv", fileEncoding="UTF-8", sep=",", header=TRUE)
train.op_SE_N.wdbc <- train.op_SE_N.wdbc[,-1]
test.op_SE_N.wdbc <- read.table("Datasets/Final/csv/test.op_SE_N.wdbc.csv", fileEncoding="UTF-8", sep=",", header=TRUE)
test.op_SE_N.wdbc <- test.op_SE_N.wdbc[,-1]


# 2) Implementación del algoritmo Naive Bayes

# Constante para la ruta de almacenamiento de gráficos
PATH_GRAPH_NB_WISC <- "Graficas/Final/Wisconsin/nb/"
PATH_RESULTS_NB_WISC <- "Resultados/Final/Wisconsin/nb/"
PATH_GRAPH_NB_COIM <- "Graficas/Final/Coimbra/nb/"
PATH_RESULTS_NB_COIM <- "Resultados/Final/Coimbra/nb/"

# Número de pliegues es igual a 10 con repeticiones de 3
train.control.nb <- trainControl(method="repeatedcv", number=10, repeats=3)

# Función para crear, entrenar y ver resultados derivados del mejor modelo de DA
process_data_to_model_nb <- function(data.train, data.test, train.cv, tipo) {
  # Inicializar variable de salida
  results <- NULL
  
  if(tipo==1){
    # Convertir variable categórica a factor
    data.train$diagnosis <- as.factor(data.train$diagnosis)
    data.test$diagnosis <- as.factor(data.test$diagnosis)
    
    # Entrenar modelo de Naive Bayes
    model.nb <- naive_bayes(diagnosis ~ ., data = data.train, usekernel = T)
  }else{
    # Convertir variable categórica a factor
    data.train$result_bc <- as.factor(data.train$result_bc)
    data.test$result_bc <- as.factor(data.test$result_bc)
    
    # Modificar los valores de los niveles de la variable result_bc
    levels(data.train$result_bc)=c("S","P")
    levels(data.test$result_bc)=c("S","P")
    
    # Entrenar modelo de Naive Bayes
    model.nb <- naive_bayes(result_bc ~ ., data = data.train, usekernel = T)
  }
  
  # Obtener los valores predecidos por el modelo de Naive Bayes
  val.pred.nb <- predict(model.nb, data.test, type = 'class')
  
  # Calcular matriz de confusión
  if(tipo==1){
    matrix.conf.nb <- confusionMatrix(val.pred.nb, data.test$diagnosis)
  }else{
    matrix.conf.nb <- confusionMatrix(val.pred.nb, data.test$result_bc)
  }
  
  # Crear variable de salida
  results$model <- model.nb
  results$data.predicted <- val.pred.nb
  results$matrix.conf <- matrix.conf.nb
  
  # Retorno de resultados
  return (results)
}

# Inicializar variable para tabla de resultados agrupados
table_result_wdbc <- NULL
val_tipo_wdbc <- c()
val_accuracy_wdbc <- c()
val_kappa_wdbc <- c()
val_sensitivity_wdbc <- c()
val_specificity_wdbc <- c()
val_f1_wdbc <- c()
val_recall_wdbc <- c()

# TODAS LAS VARIABLES SIN NORMALIZAR

# Obtención de los resultados
results.nb_comp.wdbc <- process_data_to_model_nb(train.comp.wdbc, test.comp.wdbc, train.control.nb, 1)

# Guardar valores en la tabla de resultados 
val_tipo_wdbc[1] <- "Todas las variables sin normalizar"
val_accuracy_wdbc[1] <- as.numeric(results.nb_comp.wdbc$matrix.conf$overall["Accuracy"])
val_kappa_wdbc[1] <- as.numeric(results.nb_comp.wdbc$matrix.conf$overall["Kappa"])
val_sensitivity_wdbc[1] <- as.numeric(results.nb_comp.wdbc$matrix.conf$byClass["Sensitivity"])
val_specificity_wdbc[1] <- as.numeric(results.nb_comp.wdbc$matrix.conf$byClass["Specificity"])
val_f1_wdbc[1] <- as.numeric(results.nb_comp.wdbc$matrix.conf$byClass["F1"])
val_recall_wdbc[1] <- as.numeric(results.nb_comp.wdbc$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.nb_comp.wdbc$matrix.conf$table, paste(PATH_RESULTS_NB_WISC, "matriz_confianza_nb_todas_variables_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.nb_comp.wdbc$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_NB_WISC, "matriz_confianza_nb_todas_variables_sin_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.nb_comp.wdbc$data.predicted, paste(PATH_RESULTS_NB_WISC, "datos_prediccion_nb_todas_variables_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.nb_comp.wdbc$data.predicted), type = "latex"), file = paste(PATH_RESULTS_NB_WISC, "datos_prediccion_nb_todas_variables_sin_normalizar.tex", sep=""))

# TODAS LAS VARIABLES CON NORMALIZACIÓN

# Obtención de los resultados
results.nb_comp_N.wdbc <- process_data_to_model_nb(train.comp_N.wdbc, test.comp_N.wdbc, train.control.nb, 1)

# Guardar valores en la tabla de resultados 
val_tipo_wdbc[2] <- "Todas las variables con normalizar"
val_accuracy_wdbc[2] <- as.numeric(results.nb_comp_N.wdbc$matrix.conf$overall["Accuracy"])
val_kappa_wdbc[2] <- as.numeric(results.nb_comp_N.wdbc$matrix.conf$overall["Kappa"])
val_sensitivity_wdbc[2] <- as.numeric(results.nb_comp_N.wdbc$matrix.conf$byClass["Sensitivity"])
val_specificity_wdbc[2] <- as.numeric(results.nb_comp_N.wdbc$matrix.conf$byClass["Specificity"])
val_f1_wdbc[2] <- as.numeric(results.nb_comp_N.wdbc$matrix.conf$byClass["F1"])
val_recall_wdbc[2] <- as.numeric(results.nb_comp_N.wdbc$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.nb_comp_N.wdbc$matrix.conf$table, paste(PATH_RESULTS_NB_WISC, "matriz_confianza_nb_todas_variables_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.nb_comp_N.wdbc$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_NB_WISC, "matriz_confianza_nb_todas_variables_con_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.nb_comp_N.wdbc$data.predicted, paste(PATH_RESULTS_NB_WISC, "datos_prediccion_nb_todas_variables_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.nb_comp_N.wdbc$data.predicted), type = "latex"), file = paste(PATH_RESULTS_NB_WISC, "datos_prediccion_nb_todas_variables_con_normalizar.tex", sep=""))

# VARIABLES SIN COLINEALIDAD Y SIN NORMALIZAR

# Obtención de los resultados
results.nb.op_SE.wdbc <- process_data_to_model_nb(train.op_SE.wdbc, test.op_SE.wdbc, train.control.nb, 1)

# Guardar valores en la tabla de resultados 
val_tipo_wdbc[3] <- "Variables sin colinealidad y sin normalización"
val_accuracy_wdbc[3] <- as.numeric(results.nb.op_SE.wdbc$matrix.conf$overall["Accuracy"])
val_kappa_wdbc[3] <- as.numeric(results.nb.op_SE.wdbc$matrix.conf$overall["Kappa"])
val_sensitivity_wdbc[3] <- as.numeric(results.nb.op_SE.wdbc$matrix.conf$byClass["Sensitivity"])
val_specificity_wdbc[3] <- as.numeric(results.nb.op_SE.wdbc$matrix.conf$byClass["Specificity"])
val_f1_wdbc[3] <- as.numeric(results.nb.op_SE.wdbc$matrix.conf$byClass["F1"])
val_recall_wdbc[3] <- as.numeric(results.nb.op_SE.wdbc$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.nb.op_SE.wdbc$matrix.conf$table, paste(PATH_RESULTS_NB_WISC, "matriz_confianza_nb_variables_sin_colin_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.nb.op_SE.wdbc$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_NB_WISC, "matriz_confianza_nb_variables_sin_colin_sin_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.nb.op_SE.wdbc$data.predicted, paste(PATH_RESULTS_NB_WISC, "datos_prediccion_nb_variables_sin_colin_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.nb.op_SE.wdbc$data.predicted), type = "latex"), file = paste(PATH_RESULTS_NB_WISC, "datos_prediccion_nb_variables_sin_colin_sin_normalizar.tex", sep=""))

# VARIABLES SIN COLINEALIDAD Y CON NORMALIZACIÓN

# Obtención de los resultados
results.nb.op_SE_N.wdbc <- process_data_to_model_nb(train.op_SE_N.wdbc, test.op_SE_N.wdbc, train.control.nb, 1)

# Guardar valores en la tabla de resultados 
val_tipo_wdbc[4] <- "Variables sin colinealidad y con normalización"
val_accuracy_wdbc[4] <- as.numeric(results.nb.op_SE_N.wdbc$matrix.conf$overall["Accuracy"])
val_kappa_wdbc[4] <- as.numeric(results.nb.op_SE_N.wdbc$matrix.conf$overall["Kappa"])
val_sensitivity_wdbc[4] <- as.numeric(results.nb.op_SE_N.wdbc$matrix.conf$byClass["Sensitivity"])
val_specificity_wdbc[4] <- as.numeric(results.nb.op_SE_N.wdbc$matrix.conf$byClass["Specificity"])
val_f1_wdbc[4] <- as.numeric(results.nb.op_SE_N.wdbc$matrix.conf$byClass["F1"])
val_recall_wdbc[4] <- as.numeric(results.nb.op_SE_N.wdbc$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.nb.op_SE_N.wdbc$matrix.conf$table, paste(PATH_RESULTS_NB_WISC, "matriz_confianza_nb_variables_sin_colin_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.nb.op_SE_N.wdbc$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_NB_WISC, "matriz_confianza_nb_variables_sin_colin_con_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.nb.op_SE_N.wdbc$data.predicted, paste(PATH_RESULTS_NB_WISC, "datos_prediccion_nb_variables_sin_colin_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.nb.op_SE_N.wdbc$data.predicted), type = "latex"), file = paste(PATH_RESULTS_NB_WISC, "datos_prediccion_nb_variables_sin_colin_con_normalizar.tex", sep=""))

# Fusionar columnas de resultados
table_result_wdbc <- data.frame(cbind(val_tipo_wdbc, val_accuracy_wdbc, val_kappa_wdbc, val_sensitivity_wdbc, val_specificity_wdbc, val_f1_wdbc, val_recall_wdbc))
names(table_result_wdbc) <- c('Tipo', 'Accuracy', 'Kappa', 'Sensitivity', 'Specificity', 'F1', 'Recall')

# Guardar tabla de resultados
write.csv(table_result_wdbc, paste(PATH_RESULTS_NB_WISC, "tabla_resultados_nb.csv", sep=""), row.names=TRUE)
print(xtable(table_result_wdbc, type = "latex"), file = paste(PATH_RESULTS_NB_WISC, "tabla_resultados_nb.tex", sep=""))

## CONJUNTO DE DATOS - BREAST CANCER COIMBRA DATASET

# 1) Lectura de las versiones del conjunto de datos Breast Cancer Coimbra Dataset

# 1.1) Leer datos de los archivos CSV

# Conjunto completo aplicado limpieza de datos
data_orig_coimbra <- read.table("Datasets/Final/csv/data_orig_coimbra.csv", fileEncoding="UTF-8", sep=",", header=TRUE)
data_orig_coimbra <- data_orig_coimbra[,-1]

# Conjuntos de entrenamiento y prueba conformados por todas las variables del conjunto de datos que se le aplicó la limpieza de datos
train.comp.coimbra <- read.table("Datasets/Final/csv/train.comp.coimbra.csv", fileEncoding="UTF-8", sep=",", header=TRUE)
train.comp.coimbra <- train.comp.coimbra[,-1]
test.comp.coimbra <- read.table("Datasets/Final/csv/test.comp.coimbra.csv", fileEncoding="UTF-8", sep=",", header=TRUE)
test.comp.coimbra <- test.comp.coimbra[,-1]

# Conjuntos de entrenamiento y prueba conformados por todas las variables escaladas del conjunto de datos que se le aplicó la limpieza de datos
train.comp_N.coimbra <- read.table("Datasets/Final/csv/train.comp_N.coimbraa.csv", fileEncoding="UTF-8", sep=",", header=TRUE)
train.comp_N.coimbra <- train.comp_N.coimbra[,-1]
test.comp_N.coimbra <- read.table("Datasets/Final/csv/test.comp_N.coimbra.csv", fileEncoding="UTF-8", sep=",", header=TRUE)
test.comp_N.coimbra <- test.comp_N.coimbra[,-1]

# Conjuntos de entrenamiento y prueba conformados por las variables sin colinealidad y que no se les ha aplicado el escalado de valores 
train.op_SE.coimbra <- read.table("Datasets/Final/csv/train.op_SE.coimbra.csv", fileEncoding="UTF-8", sep=",", header=TRUE)
train.op_SE.coimbra <- train.op_SE.coimbra[,-1]
test.op_SE.coimbra <- read.table("Datasets/Final/csv/test.op_SE.coimbra.csv", fileEncoding="UTF-8", sep=",", header=TRUE)
test.op_SE.coimbra <- test.op_SE.coimbra[,-1]

# Conjuntos de entrenamiento y prueba conformados por las variables sin colinealidad y que se les ha aplicado el escalado de valores
train.op_SE_N.coimbra <- read.table("Datasets/Final/csv/train.op_SE_N.coimbra.csv", fileEncoding="UTF-8", sep=",", header=TRUE)
train.op_SE_N.coimbra <- train.op_SE_N.coimbra[,-1]
test.op_SE_N.coimbra <- read.table("Datasets/Final/csv/test.op_SE_N.coimbra.csv", fileEncoding="UTF-8", sep=",", header=TRUE)
test.op_SE_N.coimbra <- test.op_SE_N.coimbra[,-1]

# 2) Implementación del algoritmo Naive Bayes

# Inicializar variable para tabla de resultados agrupados
table_result_coim <- NULL
val_tipo_coim <- c()
val_accuracy_coim <- c()
val_kappa_coim <- c()
val_sensitivity_coim <- c()
val_specificity_coim <- c()
val_f1_coim <- c()
val_recall_coim <- c()

# TODAS LAS VARIABLES SIN NORMALIZAR

# Obtención de los resultados
results.nb_comp.coim <- process_data_to_model_nb(train.comp.coimbra, test.comp.coimbra, train.control.nb, 2)

# Guardar valores en la tabla de resultados 
val_tipo_coim[1] <- "Todas las variables sin normalizar"
val_accuracy_coim[1] <- as.numeric(results.nb_comp.coim$matrix.conf$overall["Accuracy"])
val_kappa_coim[1] <- as.numeric(results.nb_comp.coim$matrix.conf$overall["Kappa"])
val_sensitivity_coim[1] <- as.numeric(results.nb_comp.coim$matrix.conf$byClass["Sensitivity"])
val_specificity_coim[1] <- as.numeric(results.nb_comp.coim$matrix.conf$byClass["Specificity"])
val_f1_coim[1] <- as.numeric(results.nb_comp.coim$matrix.conf$byClass["F1"])
val_recall_coim[1] <- as.numeric(results.nb_comp.coim$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.nb_comp.coim$matrix.conf$table, paste(PATH_RESULTS_NB_COIM, "matriz_confianza_nb_todas_variables_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.nb_comp.coim$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_NB_COIM, "matriz_confianza_nb_todas_variables_sin_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.nb_comp.coim$data.predicted, paste(PATH_RESULTS_NB_COIM, "datos_prediccion_nb_todas_variables_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.nb_comp.coim$data.predicted), type = "latex"), file = paste(PATH_RESULTS_NB_COIM, "datos_prediccion_nb_todas_variables_sin_normalizar.tex", sep=""))

# TODAS LAS VARIABLES CON NORMALIZACIÓN

# Obtención de los resultados
results.nb_comp_N.coim <- process_data_to_model_nb(train.comp_N.coimbra, test.comp_N.coimbra, train.control.nb, 2)

# Guardar valores en la tabla de resultados 
val_tipo_coim[2] <- "Todas las variables con normalizar"
val_accuracy_coim[2] <- as.numeric(results.nb_comp_N.coim$matrix.conf$overall["Accuracy"])
val_kappa_coim[2] <- as.numeric(results.nb_comp_N.coim$matrix.conf$overall["Kappa"])
val_sensitivity_coim[2] <- as.numeric(results.nb_comp_N.coim$matrix.conf$byClass["Sensitivity"])
val_specificity_coim[2] <- as.numeric(results.nb_comp_N.coim$matrix.conf$byClass["Specificity"])
val_f1_coim[2] <- as.numeric(results.nb_comp_N.coim$matrix.conf$byClass["F1"])
val_recall_coim[2] <- as.numeric(results.nb_comp_N.coim$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.nb_comp_N.coim$matrix.conf$table, paste(PATH_RESULTS_NB_COIM, "matriz_confianza_nb_todas_variables_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.nb_comp_N.coim$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_NB_COIM, "matriz_confianza_nb_todas_variables_con_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.nb_comp_N.coim$data.predicted, paste(PATH_RESULTS_NB_COIM, "datos_prediccion_nb_todas_variables_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.nb_comp_N.coim$data.predicted), type = "latex"), file = paste(PATH_RESULTS_NB_COIM, "datos_prediccion_nb_todas_variables_con_normalizar.tex", sep=""))

# VARIABLES SIN COLINEALIDAD Y SIN NORMALIZAR

# Obtención de los resultados
results.nb.op_SE.coim <- process_data_to_model_nb(train.op_SE.coimbra, test.op_SE.coimbra, train.control.nb, 2)

# Guardar valores en la tabla de resultados 
val_tipo_coim[3] <- "Variables sin colinealidad y sin normalización"
val_accuracy_coim[3] <- as.numeric(results.nb.op_SE.coim$matrix.conf$overall["Accuracy"])
val_kappa_coim[3] <- as.numeric(results.nb.op_SE.coim$matrix.conf$overall["Kappa"])
val_sensitivity_coim[3] <- as.numeric(results.nb.op_SE.coim$matrix.conf$byClass["Sensitivity"])
val_specificity_coim[3] <- as.numeric(results.nb.op_SE.coim$matrix.conf$byClass["Specificity"])
val_f1_coim[3] <- as.numeric(results.nb.op_SE.coim$matrix.conf$byClass["F1"])
val_recall_coim[3] <- as.numeric(results.nb.op_SE.coim$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.nb.op_SE.coim$matrix.conf$table, paste(PATH_RESULTS_NB_COIM, "matriz_confianza_nb_variables_sin_colin_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.nb.op_SE.coim$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_NB_COIM, "matriz_confianza_nb_variables_sin_colin_sin_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.nb.op_SE.coim$data.predicted, paste(PATH_RESULTS_NB_COIM, "datos_prediccion_nb_variables_sin_colin_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.nb.op_SE.coim$data.predicted), type = "latex"), file = paste(PATH_RESULTS_NB_COIM, "datos_prediccion_nb_variables_sin_colin_sin_normalizar.tex", sep=""))

# VARIABLES SIN COLINEALIDAD Y CON NORMALIZACIÓN

# Obtención de los resultados
results.nb.op_SE_N.coim <- process_data_to_model_nb(train.op_SE_N.coimbra, test.op_SE_N.coimbra, train.control.nb, 2)

# Guardar valores en la tabla de resultados 
val_tipo_coim[4] <- "Variables sin colinealidad y con normalización"
val_accuracy_coim[4] <- as.numeric(results.nb.op_SE_N.coim$matrix.conf$overall["Accuracy"])
val_kappa_coim[4] <- as.numeric(results.nb.op_SE_N.coim$matrix.conf$overall["Kappa"])
val_sensitivity_coim[4] <- as.numeric(results.nb.op_SE_N.coim$matrix.conf$byClass["Sensitivity"])
val_specificity_coim[4] <- as.numeric(results.nb.op_SE_N.coim$matrix.conf$byClass["Specificity"])
val_f1_coim[4] <- as.numeric(results.nb.op_SE_N.coim$matrix.conf$byClass["F1"])
val_recall_coim[4] <- as.numeric(results.nb.op_SE_N.coim$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.nb.op_SE_N.coim$matrix.conf$table, paste(PATH_RESULTS_NB_COIM, "matriz_confianza_nb_variables_sin_colin_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.nb.op_SE_N.coim$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_NB_COIM, "matriz_confianza_nb_variables_sin_colin_con_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.nb.op_SE_N.coim$data.predicted, paste(PATH_RESULTS_NB_COIM, "datos_prediccion_nb_variables_sin_colin_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.nb.op_SE_N.coim$data.predicted), type = "latex"), file = paste(PATH_RESULTS_NB_COIM, "datos_prediccion_nb_variables_sin_colin_con_normalizar.tex", sep=""))

# Fusionar columnas de resultados
table_result_coim <- data.frame(cbind(val_tipo_coim, val_accuracy_coim, val_kappa_coim, val_sensitivity_coim, val_specificity_coim, val_f1_coim, val_recall_coim))
names(table_result_coim) <- c('Tipo', 'Accuracy', 'Kappa', 'Sensitivity', 'Specificity', 'F1', 'Recall')

# Guardar tabla de resultados
write.csv(table_result_coim, paste(PATH_RESULTS_NB_COIM, "tabla_resultados_nb.csv", sep=""), row.names=TRUE)
print(xtable(table_result_coim, type = "latex"), file = paste(PATH_RESULTS_NB_COIM, "tabla_resultados_nb.tex", sep=""))

# Finalizar la medición del tiempo de ejecución
hora.final <- Sys.time()

# Obtener tiempo de ejecución en horas
tiempo.ejec <- (hora.final - hora.inicio)[[1]]/3600

# Guardar valor del tiempo de ejecución
file.connection <- file("Datasets/Final/execution_time_code_naive_bayes.txt")
writeLines(c("Tiempo de ejecución",paste(tiempo.ejec,"horas",sep=" ")), file.connection)
close(file.connection)











