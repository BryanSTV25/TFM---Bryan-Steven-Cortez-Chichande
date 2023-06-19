# Gradient Boosted Decision Trees

# Instalar librerías
install.packages("gbm")
install.packages("extrafont")
install.packages("dplyr")
install.packages("caret")
install.packages("ggplot2")
install.packages("xtable")

# Iniciar la medición del tiempo de ejecución
hora.inicio <- Sys.time()

# Llamar librerías que se van a utilizar
library("gbm")
library("extrafont")
library("dplyr")
library("caret")
library("ggplot2")
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


# 2) Implementación del algoritmo Gradient Boosted Decision Trees

# Constante para la ruta de almacenamiento de gráficos
PATH_GRAPH_GBDT_WISC <- "Graficas/Final/Wisconsin/gbdt/"
PATH_RESULTS_GBDT_WISC <- "Resultados/Final/Wisconsin/gbdt/"
PATH_GRAPH_GBDT_COIM <- "Graficas/Final/Coimbra/gbdt/"
PATH_RESULTS_GBDT_COIM <- "Resultados/Final/Coimbra/gbdt/"

# Crear grid para encontrar el mejor modelo con sus parámetros
var.grid.gbdt <- expand.grid(.n.trees=seq(100,500,by=100),.interaction.depth=seq(1,4,by=1),.shrinkage=c(.001,.01,.1), .n.minobsinnode=10)

# Número de pliegues es igual a 10 con repeticiones de 3
train.control.gbdt <- trainControl(method = "repeatedcv", number=10, repeats=3)

# Función para crear, entrenar y ver resultados derivados del mejor modelo de DA
process_data_to_model_gbdt <- function(data.train, data.test, grid, train.cv, tipo) {
  # Inicializar variable de salida
  results <- NULL
  
  if(tipo==1){
    # Convertir variable categórica a factor
    data.train$diagnosis <- as.factor(data.train$diagnosis)
    data.test$diagnosis <- as.factor(data.test$diagnosis)
    
    # Ejecutar entrenamiento de modelos para obtener el mejor modelo y sus parámetros
    train.gbdt <- train(diagnosis~.,data=data.train, method='gbm',trControl=train.cv, tuneGrid=grid)
    
    # Ejecutar modelo de Gradient Boosting
    model.gbdt <- gbm(diagnosis ~.,
                           data = data.train,
                           distribution = "multinomial",
                           shrinkage = train.gbdt$finalModel$shrinkage,
                           interaction.depth = train.gbdt$finalModel$interaction.depth,
                           n.minobsinnode = train.gbdt$finalModel$n.minobsinnode,
                           n.trees = train.gbdt$finalModel$n.trees)
  }else{
    # Convertir variable categórica a factor
    data.train$result_bc <- as.factor(data.train$result_bc)
    data.test$result_bc <- as.factor(data.test$result_bc)
    
    # Modificar los valores de los niveles de la variable result_bc
    levels(data.train$result_bc)=c("S","P")
    levels(data.test$result_bc)=c("S","P")
    
    # Ejecutar entrenamiento de modelos para obtener el mejor modelo y sus parámetros
    train.gbdt <- train(result_bc~.,data=data.train, method='gbm', trControl=train.cv, tuneGrid=grid)
    
    # Ejecutar modelo de Gradient Boosting
    model.gbdt <- gbm(result_bc ~.,
                              data = data.train,
                              distribution = "multinomial",
                              shrinkage = train.gbdt$finalModel$shrinkage,
                              interaction.depth = train.gbdt$finalModel$interaction.depth,
                              n.minobsinnode = train.gbdt$finalModel$n.minobsinnode,
                              n.trees = train.gbdt$finalModel$n.trees)
  }
  
  # Obtener predicciones con el modelo de Gradient Boosting
  val.pred.gbdt <- predict.gbm(object = model.gbdt,
                                    newdata = data.test,
                                    n.trees = train.gbdt$finalModel$n.trees,
                                    type = "response")
  
  # Verificar los resultados por las clases de la variable diagnosis
  val.class.pred <- colnames(val.pred.gbdt)[apply(val.pred.gbdt, 1, which.max)]
  
  
  #result.gbdt.wdbc <- data.frame(test.gbdt_comp.wdbc$diagnosis, var.class.names.wdbc)
  
  
  # Calcular matriz de confusión
  if(tipo==1){
    matrix.conf.gbdt <- confusionMatrix(data.test$diagnosis, as.factor(val.class.pred))
  }else{
    matrix.conf.gbdt <- confusionMatrix(data.test$result_bc, as.factor(val.class.pred))
  }
  
  # Crear variable de salida
  results$model <- model.gbdt
  results$data.predicted <- val.pred.gbdt
  results$val.class <- val.class.pred
  results$matrix.conf <- matrix.conf.gbdt
  
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
results.gbdt_comp.wdbc <- process_data_to_model_gbdt(train.comp.wdbc, test.comp.wdbc, var.grid.gbdt, train.control.gbdt, 1)

# Gráfico de la importancia de las variables
par(mar = c(5, 8, 1, 1))
png(height=717, width=1080, file=paste(PATH_GRAPH_GBDT_WISC, "grafico_importancia_variables_todas_variables_sin_escalar.png", sep=""), type = "cairo")
summary(
  results.gbdt_comp.wdbc$model, 
  cBars = 30,
  method = relative.influence,
  las = 2
)
dev.off()

# Guardar valores en la tabla de resultados 
val_tipo_wdbc[1] <- "Todas las variables sin normalizar"
val_accuracy_wdbc[1] <- as.numeric(results.gbdt_comp.wdbc$matrix.conf$overall["Accuracy"])
val_kappa_wdbc[1] <- as.numeric(results.gbdt_comp.wdbc$matrix.conf$overall["Kappa"])
val_sensitivity_wdbc[1] <- as.numeric(results.gbdt_comp.wdbc$matrix.conf$byClass["Sensitivity"])
val_specificity_wdbc[1] <- as.numeric(results.gbdt_comp.wdbc$matrix.conf$byClass["Specificity"])
val_f1_wdbc[1] <- as.numeric(results.gbdt_comp.wdbc$matrix.conf$byClass["F1"])
val_recall_wdbc[1] <- as.numeric(results.gbdt_comp.wdbc$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.gbdt_comp.wdbc$matrix.conf$table, paste(PATH_RESULTS_GBDT_WISC, "matriz_confianza_gbdt_todas_variables_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.gbdt_comp.wdbc$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_GBDT_WISC, "matriz_confianza_gbdt_todas_variables_sin_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.gbdt_comp.wdbc$data.predicted, paste(PATH_RESULTS_GBDT_WISC, "datos_prediccion_gbdt_todas_variables_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.gbdt_comp.wdbc$data.predicted), type = "latex"), file = paste(PATH_RESULTS_GBDT_WISC, "datos_prediccion_gbdt_todas_variables_sin_normalizar.tex", sep=""))


# TODAS LAS VARIABLES CON NORMALIZACIÓN

# Obtención de los resultados
results.gbdt_comp_N.wdbc <- process_data_to_model_gbdt(train.comp_N.wdbc, test.comp_N.wdbc, var.grid.gbdt, train.control.gbdt, 1)

# Gráfico de la importancia de las variables
par(mar = c(5, 8, 1, 1))
png(height=717, width=1080, file=paste(PATH_GRAPH_GBDT_WISC, "grafico_importancia_variables_todas_variables_con_escalar.png", sep=""), type = "cairo")
summary(
  results.gbdt_comp_N.wdbc$model, 
  cBars = 30,
  method = relative.influence,
  las = 2
)
dev.off()

# Guardar valores en la tabla de resultados 
val_tipo_wdbc[2] <- "Todas las variables con normalizar"
val_accuracy_wdbc[2] <- as.numeric(results.gbdt_comp_N.wdbc$matrix.conf$overall["Accuracy"])
val_kappa_wdbc[2] <- as.numeric(results.gbdt_comp_N.wdbc$matrix.conf$overall["Kappa"])
val_sensitivity_wdbc[2] <- as.numeric(results.gbdt_comp_N.wdbc$matrix.conf$byClass["Sensitivity"])
val_specificity_wdbc[2] <- as.numeric(results.gbdt_comp_N.wdbc$matrix.conf$byClass["Specificity"])
val_f1_wdbc[2] <- as.numeric(results.gbdt_comp_N.wdbc$matrix.conf$byClass["F1"])
val_recall_wdbc[2] <- as.numeric(results.gbdt_comp_N.wdbc$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.gbdt_comp_N.wdbc$matrix.conf$table, paste(PATH_RESULTS_GBDT_WISC, "matriz_confianza_gbdt_todas_variables_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.gbdt_comp_N.wdbc$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_GBDT_WISC, "matriz_confianza_gbdt_todas_variables_con_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.gbdt_comp_N.wdbc$data.predicted, paste(PATH_RESULTS_GBDT_WISC, "datos_prediccion_gbdt_todas_variables_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.gbdt_comp_N.wdbc$data.predicted), type = "latex"), file = paste(PATH_RESULTS_GBDT_WISC, "datos_prediccion_gbdt_todas_variables_con_normalizar.tex", sep=""))

# VARIABLES SIN COLINEALIDAD Y SIN NORMALIZAR

# Obtención de los resultados
results.gbdt.op_SE.wdbc <- process_data_to_model_gbdt(train.op_SE.wdbc, test.op_SE.wdbc, var.grid.gbdt, train.control.gbdt, 1)

# Gráfico de la importancia de las variables
par(mar = c(5, 8, 1, 1))
png(height=717, width=1080, file=paste(PATH_GRAPH_GBDT_WISC, "grafico_importancia_variables_sin_colin_sin_escalar.png", sep=""), type = "cairo")
summary(
  results.gbdt.op_SE.wdbc$model, 
  cBars = 30,
  method = relative.influence,
  las = 2
)
dev.off()

# Guardar valores en la tabla de resultados 
val_tipo_wdbc[3] <- "Variables sin colinealidad y sin normalización"
val_accuracy_wdbc[3] <- as.numeric(results.gbdt.op_SE.wdbc$matrix.conf$overall["Accuracy"])
val_kappa_wdbc[3] <- as.numeric(results.gbdt.op_SE.wdbc$matrix.conf$overall["Kappa"])
val_sensitivity_wdbc[3] <- as.numeric(results.gbdt.op_SE.wdbc$matrix.conf$byClass["Sensitivity"])
val_specificity_wdbc[3] <- as.numeric(results.gbdt.op_SE.wdbc$matrix.conf$byClass["Specificity"])
val_f1_wdbc[3] <- as.numeric(results.gbdt.op_SE.wdbc$matrix.conf$byClass["F1"])
val_recall_wdbc[3] <- as.numeric(results.gbdt.op_SE.wdbc$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.gbdt.op_SE.wdbc$matrix.conf$table, paste(PATH_RESULTS_GBDT_WISC, "matriz_confianza_gbdt_variables_sin_colin_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.gbdt.op_SE.wdbc$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_GBDT_WISC, "matriz_confianza_gbdt_variables_sin_colin_sin_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.gbdt.op_SE.wdbc$data.predicted, paste(PATH_RESULTS_GBDT_WISC, "datos_prediccion_gbdt_variables_sin_colin_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.gbdt.op_SE.wdbc$data.predicted), type = "latex"), file = paste(PATH_RESULTS_GBDT_WISC, "datos_prediccion_gbdt_variables_sin_colin_sin_normalizar.tex", sep=""))

# VARIABLES SIN COLINEALIDAD Y CON NORMALIZACIÓN

# Obtención de los resultados
results.gbdt.op_SE_N.wdbc <- process_data_to_model_gbdt(train.op_SE_N.wdbc, test.op_SE_N.wdbc, var.grid.gbdt, train.control.gbdt, 1)

# Gráfico de la importancia de las variables
par(mar = c(5, 8, 1, 1))
png(height=717, width=1080, file=paste(PATH_GRAPH_GBDT_WISC, "grafico_importancia_variables_sin_colin_con_escalar.png", sep=""), type = "cairo")
summary(
  results.gbdt.op_SE_N.wdbc$model, 
  cBars = 30,
  method = relative.influence,
  las = 2
)
dev.off()

# Guardar valores en la tabla de resultados 
val_tipo_wdbc[4] <- "Variables sin colinealidad y con normalización"
val_accuracy_wdbc[4] <- as.numeric(results.gbdt.op_SE_N.wdbc$matrix.conf$overall["Accuracy"])
val_kappa_wdbc[4] <- as.numeric(results.gbdt.op_SE_N.wdbc$matrix.conf$overall["Kappa"])
val_sensitivity_wdbc[4] <- as.numeric(results.gbdt.op_SE_N.wdbc$matrix.conf$byClass["Sensitivity"])
val_specificity_wdbc[4] <- as.numeric(results.gbdt.op_SE_N.wdbc$matrix.conf$byClass["Specificity"])
val_f1_wdbc[4] <- as.numeric(results.gbdt.op_SE_N.wdbc$matrix.conf$byClass["F1"])
val_recall_wdbc[4] <- as.numeric(results.gbdt.op_SE_N.wdbc$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.gbdt.op_SE_N.wdbc$matrix.conf$table, paste(PATH_RESULTS_GBDT_WISC, "matriz_confianza_gbdt_variables_sin_colin_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.gbdt.op_SE_N.wdbc$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_GBDT_WISC, "matriz_confianza_gbdt_variables_sin_colin_con_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.gbdt.op_SE_N.wdbc$data.predicted, paste(PATH_RESULTS_GBDT_WISC, "datos_prediccion_gbdt_variables_sin_colin_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.gbdt.op_SE_N.wdbc$data.predicted), type = "latex"), file = paste(PATH_RESULTS_GBDT_WISC, "datos_prediccion_gbdt_variables_sin_colin_con_normalizar.tex", sep=""))

# Fusionar columnas de resultados
table_result_wdbc <- data.frame(cbind(val_tipo_wdbc, val_accuracy_wdbc, val_kappa_wdbc, val_sensitivity_wdbc, val_specificity_wdbc, val_f1_wdbc, val_recall_wdbc))
names(table_result_wdbc) <- c('Tipo', 'Accuracy', 'Kappa', 'Sensitivity', 'Specificity', 'F1', 'Recall')

# Guardar tabla de resultados
write.csv(table_result_wdbc, paste(PATH_RESULTS_GBDT_WISC, "tabla_resultados_gbdt.csv", sep=""), row.names=TRUE)
print(xtable(table_result_wdbc, type = "latex"), file = paste(PATH_RESULTS_GBDT_WISC, "tabla_resultados_gbdt.tex", sep=""))

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

# 2) Implementación del algoritmo Gradient Boosted Decision Trees

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
results.gbdt_comp.coim <- process_data_to_model_gbdt(train.comp.coimbra, test.comp.coimbra, var.grid.gbdt, train.control.gbdt, 2)

# Gráfico de la importancia de las variables
par(mar = c(5, 8, 1, 1))
png(height=717, width=1080, file=paste(PATH_GRAPH_GBDT_COIM, "grafico_importancia_todas_variables_sin_escalar.png", sep=""), type = "cairo")
summary(
  results.gbdt_comp.coim$model, 
  cBars = 30,
  method = relative.influence,
  las = 2
)
dev.off()

# Guardar valores en la tabla de resultados 
val_tipo_coim[1] <- "Todas las variables sin normalizar"
val_accuracy_coim[1] <- as.numeric(results.gbdt_comp.coim$matrix.conf$overall["Accuracy"])
val_kappa_coim[1] <- as.numeric(results.gbdt_comp.coim$matrix.conf$overall["Kappa"])
val_sensitivity_coim[1] <- as.numeric(results.gbdt_comp.coim$matrix.conf$byClass["Sensitivity"])
val_specificity_coim[1] <- as.numeric(results.gbdt_comp.coim$matrix.conf$byClass["Specificity"])
val_f1_coim[1] <- as.numeric(results.gbdt_comp.coim$matrix.conf$byClass["F1"])
val_recall_coim[1] <- as.numeric(results.gbdt_comp.coim$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.gbdt_comp.coim$matrix.conf$table, paste(PATH_RESULTS_GBDT_COIM, "matriz_confianza_gbdt_todas_variables_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.gbdt_comp.coim$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_GBDT_COIM, "matriz_confianza_gbdt_todas_variables_sin_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.gbdt_comp.coim$data.predicted, paste(PATH_RESULTS_GBDT_COIM, "datos_prediccion_gbdt_todas_variables_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.gbdt_comp.coim$data.predicted), type = "latex"), file = paste(PATH_RESULTS_GBDT_COIM, "datos_prediccion_gbdt_todas_variables_sin_normalizar.tex", sep=""))

# TODAS LAS VARIABLES CON NORMALIZACIÓN

# Obtención de los resultados
results.gbdt_comp_N.coim <- process_data_to_model_gbdt(train.comp_N.coimbra, test.comp_N.coimbra, var.grid.gbdt, train.control.gbdt, 2)

# Gráfico de la importancia de las variables
par(mar = c(5, 8, 1, 1))
png(height=717, width=1080, file=paste(PATH_GRAPH_GBDT_COIM, "grafico_importancia_todas_variables_con_escalar.png", sep=""), type = "cairo")
summary(
  results.gbdt_comp_N.coim$model, 
  cBars = 30,
  method = relative.influence,
  las = 2
)
dev.off()

# Guardar valores en la tabla de resultados 
val_tipo_coim[2] <- "Todas las variables con normalizar"
val_accuracy_coim[2] <- as.numeric(results.gbdt_comp_N.coim$matrix.conf$overall["Accuracy"])
val_kappa_coim[2] <- as.numeric(results.gbdt_comp_N.coim$matrix.conf$overall["Kappa"])
val_sensitivity_coim[2] <- as.numeric(results.gbdt_comp_N.coim$matrix.conf$byClass["Sensitivity"])
val_specificity_coim[2] <- as.numeric(results.gbdt_comp_N.coim$matrix.conf$byClass["Specificity"])
val_f1_coim[2] <- as.numeric(results.gbdt_comp_N.coim$matrix.conf$byClass["F1"])
val_recall_coim[2] <- as.numeric(results.gbdt_comp_N.coim$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.gbdt_comp_N.coim$matrix.conf$table, paste(PATH_RESULTS_GBDT_COIM, "matriz_confianza_gbdt_todas_variables_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.gbdt_comp_N.coim$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_GBDT_COIM, "matriz_confianza_gbdt_todas_variables_con_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.gbdt_comp_N.coim$data.predicted, paste(PATH_RESULTS_GBDT_COIM, "datos_prediccion_gbdt_todas_variables_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.gbdt_comp_N.coim$data.predicted), type = "latex"), file = paste(PATH_RESULTS_GBDT_COIM, "datos_prediccion_gbdt_todas_variables_con_normalizar.tex", sep=""))

# VARIABLES SIN COLINEALIDAD Y SIN NORMALIZAR

# Obtención de los resultados
results.gbdt.op_SE.coim <- process_data_to_model_gbdt(train.op_SE.coimbra, test.op_SE.coimbra, var.grid.gbdt, train.control.gbdt, 2)

# Gráfico de la importancia de las variables
par(mar = c(5, 8, 1, 1))
png(height=717, width=1080, file=paste(PATH_GRAPH_GBDT_COIM, "grafico_importancia_variables_sin_colin_sin_escalar.png", sep=""), type = "cairo")
summary(
  results.gbdt.op_SE.coim$model, 
  cBars = 30,
  method = relative.influence,
  las = 2
)
dev.off()

# Guardar valores en la tabla de resultados 
val_tipo_coim[3] <- "Variables sin colinealidad y sin normalización"
val_accuracy_coim[3] <- as.numeric(results.gbdt.op_SE.coim$matrix.conf$overall["Accuracy"])
val_kappa_coim[3] <- as.numeric(results.gbdt.op_SE.coim$matrix.conf$overall["Kappa"])
val_sensitivity_coim[3] <- as.numeric(results.gbdt.op_SE.coim$matrix.conf$byClass["Sensitivity"])
val_specificity_coim[3] <- as.numeric(results.gbdt.op_SE.coim$matrix.conf$byClass["Specificity"])
val_f1_coim[3] <- as.numeric(results.gbdt.op_SE.coim$matrix.conf$byClass["F1"])
val_recall_coim[3] <- as.numeric(results.gbdt.op_SE.coim$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.gbdt.op_SE.coim$matrix.conf$table, paste(PATH_RESULTS_GBDT_COIM, "matriz_confianza_gbdt_variables_sin_colin_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.gbdt.op_SE.coim$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_GBDT_COIM, "matriz_confianza_gbdt_variables_sin_colin_sin_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.gbdt.op_SE.coim$data.predicted, paste(PATH_RESULTS_GBDT_COIM, "datos_prediccion_gbdt_variables_sin_colin_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.gbdt.op_SE.coim$data.predicted), type = "latex"), file = paste(PATH_RESULTS_GBDT_COIM, "datos_prediccion_gbdt_variables_sin_colin_sin_normalizar.tex", sep=""))

# VARIABLES SIN COLINEALIDAD Y CON NORMALIZACIÓN

# Obtención de los resultados
results.gbdt.op_SE_N.coim <- process_data_to_model_gbdt(train.op_SE_N.coimbra, test.op_SE_N.coimbra, var.grid.gbdt, train.control.gbdt, 2)

# Gráfico de la importancia de las variables
par(mar = c(5, 8, 1, 1))
png(height=717, width=1080, file=paste(PATH_GRAPH_GBDT_COIM, "grafico_importancia_variables_sin_colin_con_escalar.png", sep=""), type = "cairo")
summary(
  results.gbdt.op_SE_N.coim$model, 
  cBars = 30,
  method = relative.influence,
  las = 2
)
dev.off()

# Guardar valores en la tabla de resultados 
val_tipo_coim[4] <- "Variables sin colinealidad y con normalización"
val_accuracy_coim[4] <- as.numeric(results.gbdt.op_SE_N.coim$matrix.conf$overall["Accuracy"])
val_kappa_coim[4] <- as.numeric(results.gbdt.op_SE_N.coim$matrix.conf$overall["Kappa"])
val_sensitivity_coim[4] <- as.numeric(results.gbdt.op_SE_N.coim$matrix.conf$byClass["Sensitivity"])
val_specificity_coim[4] <- as.numeric(results.gbdt.op_SE_N.coim$matrix.conf$byClass["Specificity"])
val_f1_coim[4] <- as.numeric(results.gbdt.op_SE_N.coim$matrix.conf$byClass["F1"])
val_recall_coim[4] <- as.numeric(results.gbdt.op_SE_N.coim$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.gbdt.op_SE_N.coim$matrix.conf$table, paste(PATH_RESULTS_GBDT_COIM, "matriz_confianza_gbdt_variables_sin_colin_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.gbdt.op_SE_N.coim$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_GBDT_COIM, "matriz_confianza_gbdt_variables_sin_colin_con_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.gbdt.op_SE_N.coim$data.predicted, paste(PATH_RESULTS_GBDT_COIM, "datos_prediccion_gbdt_variables_sin_colin_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.gbdt.op_SE_N.coim$data.predicted), type = "latex"), file = paste(PATH_RESULTS_GBDT_COIM, "datos_prediccion_gbdt_variables_sin_colin_con_normalizar.tex", sep=""))

# Fusionar columnas de resultados
table_result_coim <- data.frame(cbind(val_tipo_coim, val_accuracy_coim, val_kappa_coim, val_sensitivity_coim, val_specificity_coim, val_f1_coim, val_recall_coim))
names(table_result_coim) <- c('Tipo', 'Accuracy', 'Kappa', 'Sensitivity', 'Specificity', 'F1', 'Recall')

# Guardar tabla de resultados
write.csv(table_result_coim, paste(PATH_RESULTS_GBDT_COIM, "tabla_resultados_gbdt.csv", sep=""), row.names=TRUE)
print(xtable(table_result_coim, type = "latex"), file = paste(PATH_RESULTS_GBDT_COIM, "tabla_resultados_gbdt.tex", sep=""))

# Finalizar la medición del tiempo de ejecución
hora.final <- Sys.time()

# Obtener tiempo de ejecución en horas
tiempo.ejec <- (hora.final - hora.inicio)[[1]]/3600

# Guardar valor del tiempo de ejecución
file.connection <- file("Datasets/Final/execution_time_code_gradient_boosted_decision_trees.txt")
writeLines(c("Tiempo de ejecución",paste(tiempo.ejec,"horas",sep=" ")), file.connection)
close(file.connection)

























