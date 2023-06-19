## k-Nearest Neighbor

# Instalar librerías
install.packages("ggplot2")
install.packages("extrafont")
install.packages("corrplot")
install.packages("caret")
install.packages("dplyr")
install.packages("xtable")

# Iniciar la medición del tiempo de ejecución
hora.inicio <- Sys.time()

# Llamar librerías que se van a utilizar
library("ggplot2")
library(extrafont)
library("corrplot")
library(caret)
library("dplyr")
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


# 2) Implementación del algoritmo k-Nearest Neighbor

# Constante para la ruta de almacenamiento de gráficos
PATH_GRAPH_KNN <- "Graficas/Final/Wisconsin/knn/"
PATH_RESULTS_KNN <- "Resultados/Final/Wisconsin/knn/"


# Convertir variable categórica a factor
data_mod_wdbc$diagnosis <- as.factor(data_mod_wdbc$diagnosis)

# Número de pliegues es igual a 10 con repeticiones de 3
train.control.knn <- trainControl(method="repeatedcv", number=10, repeats=3)

# Función para crear, entrenar y ver resultados derivados del mejor modelo de KNN
process_data_to_model_knn <- function(data.train, data.test, train.cv) {
  # Inicializar semilla
  set.seed(425)
  
  # Inicializar variable de salida
  results <- NULL
  
  # Convertir variable categórica a factor
  data.train$diagnosis <- as.factor(data.train$diagnosis)
  data.test$diagnosis <- as.factor(data.test$diagnosis)
  
  # Entrenar modelo de knn
  model.knn.wdbc <- train(diagnosis ~., data=data.train, method="knn",
                               metric="Accuracy", trControl=train.cv)
  
  # Obtener los valores predecidos por el modelo de knn
  val.pred.knn <- predict(model.knn.wdbc, newdata = data.test)
  
  # Calcular matriz de confusión
  matrix.conf.knn <- confusionMatrix(val.pred.knn, data.test$diagnosis, positive = "M")
  
  # Crear variable de salida
  results$model <- model.knn.wdbc
  results$data.predicted <- val.pred.knn
  results$matrix.conf <- matrix.conf.knn
  
  # Retorno de resultados
  return (results)
}

# Inicializar variable para tabla de resultados agrupados
table_result <- NULL
val_tipo <- c()
val_k <- c()
val_accuracy <- c()
val_kappa <- c()
val_sensitivity <- c()
val_specificity <- c()
val_f1 <- c()
val_recall <- c()


# TODAS LAS VARIABLES SIN NORMALIZAR

# Obtención de los resultados
results.knn_comp <- process_data_to_model_knn(train.comp.wdbc, test.comp.wdbc, train.control.knn)

# Graficar los valores de precisión según el valor de k
png(height=720, width=720, file=paste(PATH_GRAPH_KNN, "grafico_precision_segun_K_comp_sin_normalizar.png", sep=""), type = "cairo")
plot(results.knn_comp$model, main = "Precisión según el valor de K") 
dev.off()

# Guardar valores en la tabla de resultados 
val_tipo[1] <- "Todas las variables sin normalizar"
val_k[1] <- results.knn_comp$model$bestTune[1,1]
val_accuracy[1] <- as.numeric(results.knn_comp$matrix.conf$overall["Accuracy"])
val_kappa[1] <- as.numeric(results.knn_comp$matrix.conf$overall["Kappa"])
val_sensitivity[1] <- as.numeric(results.knn_comp$matrix.conf$byClass["Sensitivity"])
val_specificity[1] <- as.numeric(results.knn_comp$matrix.conf$byClass["Specificity"])
val_f1[1] <- as.numeric(results.knn_comp$matrix.conf$byClass["F1"])
val_recall[1] <- as.numeric(results.knn_comp$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.knn_comp$matrix.conf$table, paste(PATH_RESULTS_KNN, "matriz_confianza_knn_todas_variables_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.knn_comp$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_KNN, "matriz_confianza_knn_todas_variables_sin_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.knn_comp$data.predicted, paste(PATH_RESULTS_KNN, "datos_prediccion_knn_todas_variables_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.knn_comp$data.predicted), type = "latex"), file = paste(PATH_RESULTS_KNN, "datos_prediccion_knn_todas_variables_sin_normalizar.tex", sep=""))

# TODAS LAS VARIABLES CON NORMALIZACIÓN

# Obtención de los resultados
results.knn_comp_N <- process_data_to_model_knn(train.comp_N.wdbc, test.comp_N.wdbc, train.control.knn)

# Graficar los valores de precisión según el valor de k
png(height=720, width=720, file=paste(PATH_GRAPH_KNN, "grafico_precision_segun_K_comp_con_normalizar.png", sep=""), type = "cairo")
plot(results.knn_comp_N$model, main = "Precisión según el valor de K") 
dev.off()

# Guardar valores en la tabla de resultados 
val_tipo[2] <- "Todas las variables con normalización"
val_k[2] <- results.knn_comp_N$model$bestTune[1,1]
val_accuracy[2]  <- as.numeric(results.knn_comp_N$matrix.conf$overall["Accuracy"])
val_kappa[2] <- as.numeric(results.knn_comp_N$matrix.conf$overall["Kappa"])
val_sensitivity[2] <- as.numeric(results.knn_comp_N$matrix.conf$byClass["Sensitivity"])
val_specificity[2] <- as.numeric(results.knn_comp_N$matrix.conf$byClass["Specificity"])
val_f1[2] <- as.numeric(results.knn_comp_N$matrix.conf$byClass["F1"])
val_recall[2] <- as.numeric(results.knn_comp_N$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.knn_comp_N$matrix.conf$table, paste(PATH_RESULTS_KNN, "matriz_confianza_knn_todas_variables_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.knn_comp_N$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_KNN, "matriz_confianza_knn_todas_variables_con_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.knn_comp_N$data.predicted, paste(PATH_RESULTS_KNN, "datos_prediccion_knn_todas_variables_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.knn_comp_N$data.predicted), type = "latex"), file = paste(PATH_RESULTS_KNN, "datos_prediccion_knn_todas_variables_con_normalizar.tex", sep=""))


# VARIABLES SIN COLINEALIDAD Y SIN NORMALIZAR

# Obtención de los resultados
results.knn.op_SE <- process_data_to_model_knn(train.op_SE.wdbc, test.op_SE.wdbc, train.control.knn)

# Graficar los valores de precisión según el valor de k
png(height=720, width=720, file=paste(PATH_GRAPH_KNN, "grafico_precision_segun_K_sin_colin_sin_normalizar.png", sep=""), type = "cairo")
plot(results.knn.op_SE$model, main = "Precisión según el valor de K") 
dev.off()

# Guardar valores en la tabla de resultados 
val_tipo[3] <- "Variables sin colinealidad y sin normalización"
val_k[3] <- results.knn.op_SE$model$bestTune[1,1]
val_accuracy[3]  <- as.numeric(results.knn.op_SE$matrix.conf$overall["Accuracy"])
val_kappa[3] <- as.numeric(results.knn.op_SE$matrix.conf$overall["Kappa"])
val_sensitivity[3] <- as.numeric(results.knn.op_SE$matrix.conf$byClass["Sensitivity"])
val_specificity[3] <- as.numeric(results.knn.op_SE$matrix.conf$byClass["Specificity"])
val_f1[3] <- as.numeric(results.knn.op_SE$matrix.conf$byClass["F1"])
val_recall[3] <- as.numeric(results.knn.op_SE$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.knn.op_SE$matrix.conf$table, paste(PATH_RESULTS_KNN, "matriz_confianza_knn_variables_sin_colin_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.knn.op_SE$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_KNN, "matriz_confianza_knn_sin_colin_sin_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.knn.op_SE$data.predicted, paste(PATH_RESULTS_KNN, "datos_prediccion_knn_sin_colin_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.knn.op_SE$data.predicted), type = "latex"), file = paste(PATH_RESULTS_KNN, "datos_prediccion_knn_sin_colin_sin_normalizar.tex", sep=""))

# VARIABLES SIN COLINEALIDAD Y CON NORMALIZACIÓN

# Obtención de los resultados
results.knn.op_SE_N <- process_data_to_model_knn(train.op_SE_N.wdbc, test.op_SE_N.wdbc, train.control.knn)

# Graficar los valores de precisión según el valor de k
png(height=720, width=720, file=paste(PATH_GRAPH_KNN, "grafico_precision_segun_K_sin_colin_con_normalizar.png", sep=""), type = "cairo")
plot(results.knn.op_SE_N$model, main = "Precisión según el valor de K") 
dev.off()

# Guardar valores en la tabla de resultados 
val_tipo[4] <- "Variables sin colinealidad y con normalización"
val_k[4] <- results.knn.op_SE_N$model$bestTune[1,1]
val_accuracy[4]  <- as.numeric(results.knn.op_SE_N$matrix.conf$overall["Accuracy"])
val_kappa[4] <- as.numeric(results.knn.op_SE_N$matrix.conf$overall["Kappa"])
val_sensitivity[4] <- as.numeric(results.knn.op_SE_N$matrix.conf$byClass["Sensitivity"])
val_specificity[4] <- as.numeric(results.knn.op_SE_N$matrix.conf$byClass["Specificity"])
val_f1[4] <- as.numeric(results.knn.op_SE_N$matrix.conf$byClass["F1"])
val_recall[4] <- as.numeric(results.knn.op_SE_N$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.knn.op_SE_N$matrix.conf$table, paste(PATH_RESULTS_KNN, "matriz_confianza_knn_variables_sin_colin_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.knn.op_SE_N$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_KNN, "matriz_confianza_knn_sin_colin_con_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.knn.op_SE_N$data.predicted, paste(PATH_RESULTS_KNN, "datos_prediccion_knn_sin_colin_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.knn.op_SE_N$data.predicted), type = "latex"), file = paste(PATH_RESULTS_KNN, "datos_prediccion_knn_sin_colin_con_normalizar.tex", sep=""))

# Fusionar columnas de resultados
table_result <- data.frame(cbind(val_tipo, val_k, val_accuracy, val_kappa, val_sensitivity, val_specificity, val_f1, val_recall))
names(table_result) <- c('Tipo', 'K', 'Accuracy', 'Kappa', 'Sensitivity', 'Specificity', 'F1', 'Recall')

# Guardar tabla de resultados
write.csv(table_result, paste(PATH_RESULTS_KNN, "tabla_resultados_knn.csv", sep=""), row.names=TRUE)
print(xtable(table_result, type = "latex"), file = paste(PATH_RESULTS_KNN, "tabla_resultados_knn.tex", sep=""))

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

# 2) Implementación del algoritmo k-Nearest Neighbor

# Constante para la ruta de almacenamiento de gráficos
PATH_GRAPH_KNN_C <- "Graficas/Final/Coimbra/knn/"
PATH_RESULTS_KNN_C <- "Resultados/Final/Coimbra/knn/"

# Número de pliegues es igual a 10 con repeticiones de 3
train.control.knn_comp.coimbra <- trainControl(method="repeatedcv", number=10, repeats=3)

# Función para crear, entrenar y ver resultados derivados del mejor modelo de KNN
process_data_coimbra_to_model_knn <- function(data.train, data.test, train.cv) {
  # Inicializar variable de salida
  results <- NULL
  
  # Convertir variable categórica a factor
  data.train$result_bc <- as.factor(data.train$result_bc)
  data.test$result_bc <- as.factor(data.test$result_bc)
  
  # Modificar los valores de los niveles de la variable result_bc
  levels(data.train$result_bc)=c("S","P")
  levels(data.test$result_bc)=c("S","P")
  
  # Entrenar modelo de knn
  model.knn.coimbra <- train(result_bc ~., data=data.train, method="knn",
                          metric="Accuracy", trControl=train.cv)
  
  # Obtener los valores predecidos por el modelo de knn
  val.pred.knn <- predict(model.knn.coimbra, newdata = data.test)
  
  # Calcular matriz de confusión
  matrix.conf.knn <- confusionMatrix(val.pred.knn, data.test$result_bc, positive = "S")
  
  # Crear variable de salida
  results$model <- model.knn.coimbra
  results$data.predicted <- val.pred.knn
  results$matrix.conf <- matrix.conf.knn
  
  # Retorno de resultados
  return (results)
}

# Inicializar variable para tabla de resultados agrupados
table_result_c <- NULL
val_tipo_c <- c()
val_k_c <- c()
val_accuracy_c <- c()
val_kappa_c <- c()
val_sensitivity_c <- c()
val_specificity_c <- c()
val_f1_c <- c()
val_recall_c <- c()

# TODAS LAS VARIABLES SIN NORMALIZAR

# Obtención de los resultados
results.knn_comp.coimbra <- process_data_coimbra_to_model_knn(train.comp.coimbra, test.comp.coimbra, train.control.knn_comp.coimbra)

# Graficar los valores de precisión según el valor de k
png(height=720, width=720, file=paste(PATH_GRAPH_KNN_C, "grafico_precision_segun_K_comp_sin_normalizar.png", sep=""), type = "cairo")
plot(results.knn_comp.coimbra$model, main = "Precisión según el valor de K") 
dev.off()

# Guardar valores en la tabla de resultados 
val_tipo_c[1] <- "Todas las variables sin normalizar"
val_k_c[1] <- results.knn_comp.coimbra$model$bestTune[1,1]
val_accuracy_c[1] <- as.numeric(results.knn_comp.coimbra$matrix.conf$overall["Accuracy"])
val_kappa_c[1] <- as.numeric(results.knn_comp.coimbra$matrix.conf$overall["Kappa"])
val_sensitivity_c[1] <- as.numeric(results.knn_comp.coimbra$matrix.conf$byClass["Sensitivity"])
val_specificity_c[1] <- as.numeric(results.knn_comp.coimbra$matrix.conf$byClass["Specificity"])
val_f1_c[1] <- as.numeric(results.knn_comp.coimbra$matrix.conf$byClass["F1"])
val_recall_c[1] <- as.numeric(results.knn_comp.coimbra$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.knn_comp.coimbra$matrix.conf$table, paste(PATH_RESULTS_KNN_C, "matriz_confianza_knn_todas_variables_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.knn_comp.coimbra$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_KNN_C, "matriz_confianza_knn_todas_variables_sin_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.knn_comp.coimbra$data.predicted, paste(PATH_RESULTS_KNN_C, "datos_prediccion_knn_todas_variables_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.knn_comp.coimbra$data.predicted), type = "latex"), file = paste(PATH_RESULTS_KNN_C, "datos_prediccion_knn_todas_variables_sin_normalizar.tex", sep=""))

# TODAS LAS VARIABLES CON NORMALIZACIÓN

# Obtención de los resultados
results.knn_comp_N.coimbra <- process_data_coimbra_to_model_knn(train.comp_N.coimbra, test.comp_N.coimbra, train.control.knn_comp.coimbra)

# Graficar los valores de precisión según el valor de k
png(height=720, width=720, file=paste(PATH_GRAPH_KNN_C, "grafico_precision_segun_K_comp_con_normalizar.png", sep=""), type = "cairo")
plot(results.knn_comp_N.coimbra$model, main = "Precisión según el valor de K") 
dev.off()

# Guardar valores en la tabla de resultados 
val_tipo_c[2] <- "Todas las variables con normalización"
val_k_c[2] <- results.knn_comp_N.coimbra$model$bestTune[1,1]
val_accuracy_c[2]  <- as.numeric(results.knn_comp_N.coimbra$matrix.conf$overall["Accuracy"])
val_kappa_c[2] <- as.numeric(results.knn_comp_N.coimbra$matrix.conf$overall["Kappa"])
val_sensitivity_c[2] <- as.numeric(results.knn_comp_N.coimbra$matrix.conf$byClass["Sensitivity"])
val_specificity_c[2] <- as.numeric(results.knn_comp_N.coimbra$matrix.conf$byClass["Specificity"])
val_f1_c[2] <- as.numeric(results.knn_comp_N.coimbra$matrix.conf$byClass["F1"])
val_recall_c[2] <- as.numeric(results.knn_comp_N.coimbra$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.knn_comp_N.coimbra$matrix.conf$table, paste(PATH_RESULTS_KNN_C, "matriz_confianza_knn_todas_variables_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.knn_comp_N.coimbra$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_KNN_C, "matriz_confianza_knn_todas_variables_con_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.knn_comp_N.coimbra$data.predicted, paste(PATH_RESULTS_KNN_C, "datos_prediccion_knn_todas_variables_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.knn_comp_N.coimbra$data.predicted), type = "latex"), file = paste(PATH_RESULTS_KNN_C, "datos_prediccion_knn_todas_variables_con_normalizar.tex", sep=""))

# VARIABLES SIN COLINEALIDAD Y SIN NORMALIZAR

# Obtención de los resultados
results.knn.op_SE.coimbra <- process_data_coimbra_to_model_knn(train.op_SE.coimbra, test.op_SE.coimbra, train.control.knn_comp.coimbra)

# Graficar los valores de precisión según el valor de k
png(height=720, width=720, file=paste(PATH_GRAPH_KNN_C, "grafico_precision_segun_K_sin_colin_sin_normalizar.png", sep=""), type = "cairo")
plot(results.knn.op_SE.coimbra$model, main = "Precisión según el valor de K") 
dev.off()

# Guardar valores en la tabla de resultados 
val_tipo_c[3] <- "Variables sin colinealidad y sin normalización"
val_k_c[3] <- results.knn.op_SE.coimbra$model$bestTune[1,1]
val_accuracy_c[3]  <- as.numeric(results.knn.op_SE.coimbra$matrix.conf$overall["Accuracy"])
val_kappa_c[3] <- as.numeric(results.knn.op_SE.coimbra$matrix.conf$overall["Kappa"])
val_sensitivity_c[3] <- as.numeric(results.knn.op_SE.coimbra$matrix.conf$byClass["Sensitivity"])
val_specificity_c[3] <- as.numeric(results.knn.op_SE.coimbra$matrix.conf$byClass["Specificity"])
val_f1_c[3] <- as.numeric(results.knn.op_SE.coimbra$matrix.conf$byClass["F1"])
val_recall_c[3] <- as.numeric(results.knn.op_SE.coimbra$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.knn.op_SE.coimbra$matrix.conf$table, paste(PATH_RESULTS_KNN_C, "matriz_confianza_knn_variables_sin_colin_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.knn.op_SE.coimbra$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_KNN_C, "matriz_confianza_knn_sin_colin_sin_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.knn.op_SE.coimbra$data.predicted, paste(PATH_RESULTS_KNN_C, "datos_prediccion_knn_sin_colin_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.knn.op_SE.coimbra$data.predicted), type = "latex"), file = paste(PATH_RESULTS_KNN_C, "datos_prediccion_knn_sin_colin_sin_normalizar.tex", sep=""))

# VARIABLES SIN COLINEALIDAD Y CON NORMALIZACIÓN

# Obtención de los resultados
results.knn.op_SE_N.coimbra <- process_data_coimbra_to_model_knn(train.op_SE_N.coimbra, test.op_SE_N.coimbra, train.control.knn_comp.coimbra)

# Graficar los valores de precisión según el valor de k
png(height=720, width=720, file=paste(PATH_GRAPH_KNN_C, "grafico_precision_segun_K_sin_colin_con_normalizar.png", sep=""), type = "cairo")
plot(results.knn.op_SE_N.coimbra$model, main = "Precisión según el valor de K") 
dev.off()

# Guardar valores en la tabla de resultados 
val_tipo_c[4] <- "Variables sin colinealidad y con normalización"
val_k_c[4] <- results.knn.op_SE_N.coimbra$model$bestTune[1,1]
val_accuracy_c[4]  <- as.numeric(results.knn.op_SE_N.coimbra$matrix.conf$overall["Accuracy"])
val_kappa_c[4] <- as.numeric(results.knn.op_SE_N.coimbra$matrix.conf$overall["Kappa"])
val_sensitivity_c[4] <- as.numeric(results.knn.op_SE_N.coimbra$matrix.conf$byClass["Sensitivity"])
val_specificity_c[4] <- as.numeric(results.knn.op_SE_N.coimbra$matrix.conf$byClass["Specificity"])
val_f1_c[4] <- as.numeric(results.knn.op_SE_N.coimbra$matrix.conf$byClass["F1"])
val_recall_c[4] <- as.numeric(results.knn.op_SE_N.coimbra$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.knn.op_SE_N.coimbra$matrix.conf$table, paste(PATH_RESULTS_KNN_C, "matriz_confianza_knn_variables_sin_colin_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.knn.op_SE_N.coimbra$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_KNN_C, "matriz_confianza_knn_sin_colin_con_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.knn.op_SE_N.coimbra$data.predicted, paste(PATH_RESULTS_KNN_C, "datos_prediccion_knn_sin_colin_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.knn.op_SE_N.coimbra$data.predicted), type = "latex"), file = paste(PATH_RESULTS_KNN_C, "datos_prediccion_knn_sin_colin_con_normalizar.tex", sep=""))

# Fusionar columnas de resultados
table_result_c <- data.frame(cbind(val_tipo_c, val_k_c, val_accuracy_c, val_kappa_c, val_sensitivity_c, val_specificity_c, val_f1_c, val_recall_c))
names(table_result_c) <- c('Tipo', 'K', 'Accuracy', 'Kappa', 'Sensitivity', 'Specificity', 'F1', 'Recall')

# Guardar tabla de resultados
write.csv(table_result_c, paste(PATH_RESULTS_KNN_C, "tabla_resultados_knn.csv", sep=""), row.names=TRUE)
print(xtable(table_result_c, type = "latex"), file = paste(PATH_RESULTS_KNN_C, "tabla_resultados_knn.tex", sep=""))

# Finalizar la medición del tiempo de ejecución
hora.final <- Sys.time()

# Obtener tiempo de ejecución en horas
tiempo.ejec <- (hora.final - hora.inicio)[[1]]/3600

# Guardar valor del tiempo de ejecución
file.connection <- file("Datasets/Final/execution_time_code_k_nearest_neighbor.txt")
writeLines(c("Tiempo de ejecución",paste(tiempo.ejec,"horas",sep=" ")), file.connection)
close(file.connection)






















