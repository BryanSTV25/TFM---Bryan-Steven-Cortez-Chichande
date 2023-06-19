## Logistic regression

# Instalar librerías
install.packages("ggplot2")
install.packages("extrafont")
install.packages("corrplot")
install.packages("caret")
install.packages("dplyr")
install.packages("pROC")
install.packages("xtable")

# Iniciar la medición del tiempo de ejecución
hora.inicio <- Sys.time()

# Llamar librerías que se van a utilizar
library("ggplot2")
library(extrafont)
library("corrplot")
library(caret)
library("dplyr")
library(pROC)
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


# 2) Implementación del algoritmo Logistic regression

# Constante para la ruta de almacenamiento de gráficos
PATH_GRAPH_GLM_WISC <- "Graficas/Final/Wisconsin/glm/"
PATH_RESULTS_GLM_WISC <- "Resultados/Final/Wisconsin/glm/"
PATH_GRAPH_GLM_COIM <- "Graficas/Final/Coimbra/glm/"
PATH_RESULTS_GLM_COIM <- "Resultados/Final/Coimbra/glm/"

# Número de pliegues es igual a 10 con repeticiones de 3
train.control.glm <- trainControl(method="cv",
                                       number = 10,
                                       classProbs = TRUE,
                                       savePredictions="all",
                                       summaryFunction = twoClassSummary)

# Función para crear, entrenar y ver resultados derivados del mejor modelo de GLM
process_data_to_model_glm <- function(data.train, data.test, train.cv, tipo_dt) {
  # Inicializar variable de salida
  results <- NULL
  
  if(tipo_dt==1){
    # Convertir variable categórica a factor
    data.train$diagnosis <- as.factor(data.train$diagnosis)
    data.test$diagnosis <- as.factor(data.test$diagnosis)
    
    # Entrenar modelo de knn
    model.glm <- train(diagnosis ~., data = data.train, method = "glm", 
                       metric = "ROC", trControl = train.cv)
  }else{
    # Convertir variable categórica a factor
    data.train$result_bc <- as.factor(data.train$result_bc)
    data.test$result_bc <- as.factor(data.test$result_bc)
    
    # Modificar los valores de los niveles de la variable result_bc
    levels(data.train$result_bc)=c("S","P")
    levels(data.test$result_bc)=c("S","P")
    
    # Entrenar modelo de knn
    model.glm <- train(result_bc ~., data = data.train, method = "glm", 
                       metric = "ROC", trControl = train.cv)
  }
  
  # Obtener los valores predecidos por el modelo de knn
  val.pred.glm <- predict(model.glm, newdata = data.test)
  if(tipo_dt==1){
    # Calcular matriz de confusión
    matrix.conf.glm <- confusionMatrix(val.pred.glm, data.test$diagnosis, positive = "M")
    
    # Cálculo de la curva ROC
    val.roc.glm <- roc(as.numeric(data.test$diagnosis), as.numeric(val.pred.glm))
    
    # Cálculo de AUC
    val.auc.glm <- round(auc(as.numeric(data.test$diagnosis), as.numeric(val.pred.glm)),4)
  }else{
    # Calcular matriz de confusión
    matrix.conf.glm <- confusionMatrix(val.pred.glm, data.test$result_bc, positive = "S")
    
    # Cálculo de la curva ROC
    val.roc.glm <- roc(as.numeric(data.test$result_bc), as.numeric(val.pred.glm))
    
    # Cálculo de AUC
    val.auc.glm <- round(auc(as.numeric(data.test$result_bc), as.numeric(val.pred.glm)),4)
  }
  
  # Crear variable de salida
  results$model <- model.glm
  results$data.predicted <- val.pred.glm
  results$matrix.conf <- matrix.conf.glm
  results$roc <- val.roc.glm
  results$auc <- val.auc.glm
  
  # Retorno de resultados
  return (results)
}

# Inicializar variable para tabla de resultados agrupados
table_result_wdbc <- NULL
val_tipo_wdbc <- c()
val_auc_wdbc <- c()
val_accuracy_wdbc <- c()
val_kappa_wdbc <- c()
val_sensitivity_wdbc <- c()
val_specificity_wdbc <- c()
val_f1_wdbc <- c()
val_recall_wdbc <- c()

# TODAS LAS VARIABLES SIN NORMALIZAR

# Obtención de los resultados
results.glm_comp.wdbc <- process_data_to_model_glm(train.comp.wdbc, test.comp.wdbc, train.control.glm, 1)

# Graficar curva ROC
ggroc(results.glm_comp.wdbc$roc, colour = 'steelblue', size = 2) +
  ggtitle(paste0('Curva ROC ', '(AUC = ', results.glm_comp.wdbc$auc, ') - ', 'Regresión logística')) +
  theme_classic()+
  theme(plot.title=element_text(hjust=0.5), plot.subtitle=element_text(hjust=0.5), 
        text=element_text(size=14, family="Arial Narrow"))

# Guardar gráfica de curva ROC
ggsave(path = PATH_GRAPH_GLM_WISC, filename = "curva_roc_modelo_glm_comp_sin_normalizar.png", 
       width = 1800, height = 1800, units = "px")

# Guardar valores en la tabla de resultados 
val_tipo_wdbc[1] <- "Todas las variables sin normalizar"
val_auc_wdbc[1] <- results.glm_comp.wdbc$auc
val_accuracy_wdbc[1] <- as.numeric(results.glm_comp.wdbc$matrix.conf$overall["Accuracy"])
val_kappa_wdbc[1] <- as.numeric(results.glm_comp.wdbc$matrix.conf$overall["Kappa"])
val_sensitivity_wdbc[1] <- as.numeric(results.glm_comp.wdbc$matrix.conf$byClass["Sensitivity"])
val_specificity_wdbc[1] <- as.numeric(results.glm_comp.wdbc$matrix.conf$byClass["Specificity"])
val_f1_wdbc[1] <- as.numeric(results.glm_comp.wdbc$matrix.conf$byClass["F1"])
val_recall_wdbc[1] <- as.numeric(results.glm_comp.wdbc$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.glm_comp.wdbc$matrix.conf$table, paste(PATH_RESULTS_GLM_WISC, "matriz_confianza_glm_todas_variables_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.glm_comp.wdbc$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_GLM_WISC, "matriz_confianza_glm_todas_variables_sin_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.glm_comp.wdbc$data.predicted, paste(PATH_RESULTS_GLM_WISC, "datos_prediccion_glm_todas_variables_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.glm_comp.wdbc$data.predicted), type = "latex"), file = paste(PATH_RESULTS_GLM_WISC, "datos_prediccion_glm_todas_variables_sin_normalizar.tex", sep=""))

# TODAS LAS VARIABLES CON NORMALIZACIÓN

# Obtención de los resultados
results.glm_comp_N <- process_data_to_model_glm(train.comp_N.wdbc, test.comp_N.wdbc, train.control.glm, 1)

# Graficar curva ROC
ggroc(results.glm_comp_N$roc, colour = 'steelblue', size = 2) +
  ggtitle(paste0('Curva ROC ', '(AUC = ', results.glm_comp_N$auc, ') - ', 'Regresión logística')) +
  theme_classic()+
  theme(plot.title=element_text(hjust=0.5), plot.subtitle=element_text(hjust=0.5), 
        text=element_text(size=14, family="Arial Narrow"))

# Guardar gráfica de curva ROC
ggsave(path = PATH_GRAPH_GLM_WISC, filename = "curva_roc_modelo_glm_comp_con_normalizar.png", 
       width = 1800, height = 1800, units = "px")

# Guardar valores en la tabla de resultados 
val_tipo_wdbc[2] <- "Todas las variables con normalizar"
val_auc_wdbc[2] <- results.glm_comp_N$auc
val_accuracy_wdbc[2] <- as.numeric(results.glm_comp_N$matrix.conf$overall["Accuracy"])
val_kappa_wdbc[2] <- as.numeric(results.glm_comp_N$matrix.conf$overall["Kappa"])
val_sensitivity_wdbc[2] <- as.numeric(results.glm_comp_N$matrix.conf$byClass["Sensitivity"])
val_specificity_wdbc[2] <- as.numeric(results.glm_comp_N$matrix.conf$byClass["Specificity"])
val_f1_wdbc[2] <- as.numeric(results.glm_comp_N$matrix.conf$byClass["F1"])
val_recall_wdbc[2] <- as.numeric(results.glm_comp_N$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.glm_comp_N$matrix.conf$table, paste(PATH_RESULTS_GLM_WISC, "matriz_confianza_glm_todas_variables_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.glm_comp_N$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_GLM_WISC, "matriz_confianza_glm_todas_variables_con_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.glm_comp_N$data.predicted, paste(PATH_RESULTS_GLM_WISC, "datos_prediccion_glm_todas_variables_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.glm_comp_N$data.predicted), type = "latex"), file = paste(PATH_RESULTS_GLM_WISC, "datos_prediccion_glm_todas_variables_con_normalizar.tex", sep=""))

# VARIABLES SIN COLINEALIDAD Y SIN NORMALIZAR

# Obtención de los resultados
results.glm.op_SE.wdbc <- process_data_to_model_glm(train.op_SE.wdbc, test.op_SE.wdbc, train.control.glm, 1)

# Graficar curva ROC
ggroc(results.glm.op_SE.wdbc$roc, colour = 'steelblue', size = 2) +
  ggtitle(paste0('Curva ROC ', '(AUC = ', results.glm.op_SE.wdbc$auc, ') - ', 'Regresión logística')) +
  theme_classic()+
  theme(plot.title=element_text(hjust=0.5), plot.subtitle=element_text(hjust=0.5), 
        text=element_text(size=14, family="Arial Narrow"))

# Guardar gráfica de curva ROC
ggsave(path = PATH_GRAPH_GLM_WISC, filename = "curva_roc_modelo_glm_sin_colin_sin_normalizar.png", 
       width = 1800, height = 1800, units = "px")

# Guardar valores en la tabla de resultados 
val_tipo_wdbc[3] <- "Variables sin colinealidad y sin normalización"
val_auc_wdbc[3] <- results.glm.op_SE.wdbc$auc
val_accuracy_wdbc[3] <- as.numeric(results.glm.op_SE.wdbc$matrix.conf$overall["Accuracy"])
val_kappa_wdbc[3] <- as.numeric(results.glm.op_SE.wdbc$matrix.conf$overall["Kappa"])
val_sensitivity_wdbc[3] <- as.numeric(results.glm.op_SE.wdbc$matrix.conf$byClass["Sensitivity"])
val_specificity_wdbc[3] <- as.numeric(results.glm.op_SE.wdbc$matrix.conf$byClass["Specificity"])
val_f1_wdbc[3] <- as.numeric(results.glm.op_SE.wdbc$matrix.conf$byClass["F1"])
val_recall_wdbc[3] <- as.numeric(results.glm.op_SE.wdbc$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.glm.op_SE.wdbc$matrix.conf$table, paste(PATH_RESULTS_GLM_WISC, "matriz_confianza_glm_variables_sin_colin_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.glm.op_SE.wdbc$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_GLM_WISC, "matriz_confianza_glm_variables_sin_colin_sin_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.glm.op_SE.wdbc$data.predicted, paste(PATH_RESULTS_GLM_WISC, "datos_prediccion_glm_variables_sin_colin_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.glm.op_SE.wdbc$data.predicted), type = "latex"), file = paste(PATH_RESULTS_GLM_WISC, "datos_prediccion_glm_variables_sin_colin_sin_normalizar.tex", sep=""))

# VARIABLES SIN COLINEALIDAD Y CON NORMALIZACIÓN

# Obtención de los resultados
results.glm.op_SE_N.wdbc <- process_data_to_model_glm(train.op_SE_N.wdbc, test.op_SE_N.wdbc, train.control.glm, 1)

# Graficar curva ROC
ggroc(results.glm.op_SE_N.wdbc$roc, colour = 'steelblue', size = 2) +
  ggtitle(paste0('Curva ROC ', '(AUC = ', results.glm.op_SE_N.wdbc$auc, ') - ', 'Regresión logística')) +
  theme_classic()+
  theme(plot.title=element_text(hjust=0.5), plot.subtitle=element_text(hjust=0.5), 
        text=element_text(size=14, family="Arial Narrow"))

# Guardar gráfica de curva ROC
ggsave(path = PATH_GRAPH_GLM_WISC, filename = "curva_roc_modelo_glm_sin_colin_con_normalizar.png", 
       width = 1800, height = 1800, units = "px")

# Guardar valores en la tabla de resultados 
val_tipo_wdbc[4] <- "Variables sin colinealidad y con normalización"
val_auc_wdbc[4] <- results.glm.op_SE_N.wdbc$auc
val_accuracy_wdbc[4] <- as.numeric(results.glm.op_SE_N.wdbc$matrix.conf$overall["Accuracy"])
val_kappa_wdbc[4] <- as.numeric(results.glm.op_SE_N.wdbc$matrix.conf$overall["Kappa"])
val_sensitivity_wdbc[4] <- as.numeric(results.glm.op_SE_N.wdbc$matrix.conf$byClass["Sensitivity"])
val_specificity_wdbc[4] <- as.numeric(results.glm.op_SE_N.wdbc$matrix.conf$byClass["Specificity"])
val_f1_wdbc[4] <- as.numeric(results.glm.op_SE_N.wdbc$matrix.conf$byClass["F1"])
val_recall_wdbc[4] <- as.numeric(results.glm.op_SE_N.wdbc$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.glm.op_SE_N.wdbc$matrix.conf$table, paste(PATH_RESULTS_GLM_WISC, "matriz_confianza_glm_variables_sin_colin_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.glm.op_SE_N.wdbc$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_GLM_WISC, "matriz_confianza_glm_variables_sin_colin_con_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.glm.op_SE_N.wdbc$data.predicted, paste(PATH_RESULTS_GLM_WISC, "datos_prediccion_glm_variables_sin_colin_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.glm.op_SE_N.wdbc$data.predicted), type = "latex"), file = paste(PATH_RESULTS_GLM_WISC, "datos_prediccion_glm_variables_sin_colin_con_normalizar.tex", sep=""))

# Fusionar columnas de resultados
table_result_wdbc <- data.frame(cbind(val_tipo_wdbc, val_auc_wdbc, val_accuracy_wdbc, val_kappa_wdbc, val_sensitivity_wdbc, val_specificity_wdbc, val_f1_wdbc, val_recall_wdbc))
names(table_result_wdbc) <- c('Tipo', 'AUC', 'Accuracy', 'Kappa', 'Sensitivity', 'Specificity', 'F1', 'Recall')

# Guardar tabla de resultados
write.csv(table_result_wdbc, paste(PATH_RESULTS_GLM_WISC, "tabla_resultados_glm.csv", sep=""), row.names=TRUE)
print(xtable(table_result_wdbc, type = "latex"), file = paste(PATH_RESULTS_GLM_WISC, "tabla_resultados_glm.tex", sep=""))

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

# 2) Implementación del algoritmo Logistic regression

# Inicializar variable para tabla de resultados agrupados
table_result_coim <- NULL
val_tipo_coim <- c()
val_auc_coim <- c()
val_accuracy_coim <- c()
val_kappa_coim <- c()
val_sensitivity_coim <- c()
val_specificity_coim <- c()
val_f1_coim <- c()
val_recall_coim <- c()

# TODAS LAS VARIABLES SIN NORMALIZAR

# Obtención de los resultados
results.glm_comp.coim <- process_data_to_model_glm(train.comp.coimbra, test.comp.coimbra, train.control.glm, 2)

# Graficar curva ROC
ggroc(results.glm_comp.coim$roc, colour = 'steelblue', size = 2) +
  ggtitle(paste0('Curva ROC ', '(AUC = ', results.glm_comp.coim$auc, ') - ', 'Regresión logística')) +
  theme_classic()+
  theme(plot.title=element_text(hjust=0.5), plot.subtitle=element_text(hjust=0.5), 
        text=element_text(size=14, family="Arial Narrow"))

# Guardar gráfica de curva ROC
ggsave(path = PATH_GRAPH_GLM_COIM, filename = "curva_roc_modelo_glm_comp_sin_normalizar.png", 
       width = 1800, height = 1800, units = "px")

# Guardar valores en la tabla de resultados 
val_tipo_coim[1] <- "Todas las variables sin normalizar"
val_auc_coim[1] <- results.glm_comp.coim$auc
val_accuracy_coim[1] <- as.numeric(results.glm_comp.coim$matrix.conf$overall["Accuracy"])
val_kappa_coim[1] <- as.numeric(results.glm_comp.coim$matrix.conf$overall["Kappa"])
val_sensitivity_coim[1] <- as.numeric(results.glm_comp.coim$matrix.conf$byClass["Sensitivity"])
val_specificity_coim[1] <- as.numeric(results.glm_comp.coim$matrix.conf$byClass["Specificity"])
val_f1_coim[1] <- as.numeric(results.glm_comp.coim$matrix.conf$byClass["F1"])
val_recall_coim[1] <- as.numeric(results.glm_comp.coim$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.glm_comp.coim$matrix.conf$table, paste(PATH_RESULTS_GLM_COIM, "matriz_confianza_glm_todas_variables_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.glm_comp.coim$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_GLM_COIM, "matriz_confianza_glm_todas_variables_sin_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.glm_comp.coim$data.predicted, paste(PATH_RESULTS_GLM_COIM, "datos_prediccion_glm_todas_variables_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.glm_comp.coim$data.predicted), type = "latex"), file = paste(PATH_RESULTS_GLM_COIM, "datos_prediccion_glm_todas_variables_sin_normalizar.tex", sep=""))

# TODAS LAS VARIABLES CON NORMALIZACIÓN

# Obtención de los resultados
results.glm_comp_N.coim <- process_data_to_model_glm(train.comp_N.coimbra, test.comp_N.coimbra, train.control.glm, 2)

# Graficar curva ROC
ggroc(results.glm_comp_N.coim$roc, colour = 'steelblue', size = 2) +
  ggtitle(paste0('Curva ROC ', '(AUC = ', results.glm_comp_N.coim$auc, ') - ', 'Regresión logística')) +
  theme_classic()+
  theme(plot.title=element_text(hjust=0.5), plot.subtitle=element_text(hjust=0.5), 
        text=element_text(size=14, family="Arial Narrow"))

# Guardar gráfica de curva ROC
ggsave(path = PATH_GRAPH_GLM_COIM, filename = "curva_roc_modelo_glm_comp_con_normalizar.png", 
       width = 1800, height = 1800, units = "px")

# Guardar valores en la tabla de resultados 
val_tipo_coim[2] <- "Todas las variables con normalizar"
val_auc_coim[2] <- results.glm_comp_N.coim$auc
val_accuracy_coim[2] <- as.numeric(results.glm_comp_N.coim$matrix.conf$overall["Accuracy"])
val_kappa_coim[2] <- as.numeric(results.glm_comp_N.coim$matrix.conf$overall["Kappa"])
val_sensitivity_coim[2] <- as.numeric(results.glm_comp_N.coim$matrix.conf$byClass["Sensitivity"])
val_specificity_coim[2] <- as.numeric(results.glm_comp_N.coim$matrix.conf$byClass["Specificity"])
val_f1_coim[2] <- as.numeric(results.glm_comp_N.coim$matrix.conf$byClass["F1"])
val_recall_coim[2] <- as.numeric(results.glm_comp_N.coim$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.glm_comp_N.coim$matrix.conf$table, paste(PATH_RESULTS_GLM_COIM, "matriz_confianza_glm_todas_variables_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.glm_comp_N.coim$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_GLM_COIM, "matriz_confianza_glm_todas_variables_con_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.glm_comp_N.coim$data.predicted, paste(PATH_RESULTS_GLM_COIM, "datos_prediccion_glm_todas_variables_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.glm_comp_N.coim$data.predicted), type = "latex"), file = paste(PATH_RESULTS_GLM_COIM, "datos_prediccion_glm_todas_variables_con_normalizar.tex", sep=""))

# VARIABLES SIN COLINEALIDAD Y SIN NORMALIZAR

# Obtención de los resultados
results.glm.op_SE.coim <- process_data_to_model_glm(train.op_SE.coimbra, test.op_SE.coimbra, train.control.glm, 2)

# Graficar curva ROC
ggroc(results.glm.op_SE.coim$roc, colour = 'steelblue', size = 2) +
  ggtitle(paste0('Curva ROC ', '(AUC = ', results.glm.op_SE.coim$auc, ') - ', 'Regresión logística')) +
  theme_classic()+
  theme(plot.title=element_text(hjust=0.5), plot.subtitle=element_text(hjust=0.5), 
        text=element_text(size=14, family="Arial Narrow"))

# Guardar gráfica de curva ROC
ggsave(path = PATH_GRAPH_GLM_COIM, filename = "curva_roc_modelo_glm_sin_colin_sin_normalizar.png", 
       width = 1800, height = 1800, units = "px")

# Guardar valores en la tabla de resultados 
val_tipo_coim[3] <- "Variables sin colinealidad y sin normalización"
val_auc_coim[3] <- results.glm.op_SE.coim$auc
val_accuracy_coim[3] <- as.numeric(results.glm.op_SE.coim$matrix.conf$overall["Accuracy"])
val_kappa_coim[3] <- as.numeric(results.glm.op_SE.coim$matrix.conf$overall["Kappa"])
val_sensitivity_coim[3] <- as.numeric(results.glm.op_SE.coim$matrix.conf$byClass["Sensitivity"])
val_specificity_coim[3] <- as.numeric(results.glm.op_SE.coim$matrix.conf$byClass["Specificity"])
val_f1_coim[3] <- as.numeric(results.glm.op_SE.coim$matrix.conf$byClass["F1"])
val_recall_coim[3] <- as.numeric(results.glm.op_SE.coim$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.glm.op_SE.coim$matrix.conf$table, paste(PATH_RESULTS_GLM_COIM, "matriz_confianza_glm_variables_sin_colin_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.glm.op_SE.coim$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_GLM_COIM, "matriz_confianza_glm_variables_sin_colin_sin_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.glm.op_SE.coim$data.predicted, paste(PATH_RESULTS_GLM_COIM, "datos_prediccion_glm_variables_sin_colin_sin_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.glm.op_SE.coim$data.predicted), type = "latex"), file = paste(PATH_RESULTS_GLM_COIM, "datos_prediccion_glm_variables_sin_colin_sin_normalizar.tex", sep=""))

# VARIABLES SIN COLINEALIDAD Y CON NORMALIZACIÓN

# Obtención de los resultados
results.glm.op_SE_N.coim <- process_data_to_model_glm(train.op_SE_N.coimbra, test.op_SE_N.coimbra, train.control.glm, 2)

# Graficar curva ROC
ggroc(results.glm.op_SE_N.coim$roc, colour = 'steelblue', size = 2) +
  ggtitle(paste0('Curva ROC ', '(AUC = ', results.glm.op_SE_N.coim$auc, ') - ', 'Regresión logística')) +
  theme_classic()+
  theme(plot.title=element_text(hjust=0.5), plot.subtitle=element_text(hjust=0.5), 
        text=element_text(size=14, family="Arial Narrow"))

# Guardar gráfica de curva ROC
ggsave(path = PATH_GRAPH_GLM_COIM, filename = "curva_roc_modelo_glm_sin_colin_con_normalizar.png", 
       width = 1800, height = 1800, units = "px")

# Guardar valores en la tabla de resultados 
val_tipo_coim[4] <- "Variables sin colinealidad y con normalización"
val_auc_coim[4] <- results.glm.op_SE_N.coim$auc
val_accuracy_coim[4] <- as.numeric(results.glm.op_SE_N.coim$matrix.conf$overall["Accuracy"])
val_kappa_coim[4] <- as.numeric(results.glm.op_SE_N.coim$matrix.conf$overall["Kappa"])
val_sensitivity_coim[4] <- as.numeric(results.glm.op_SE_N.coim$matrix.conf$byClass["Sensitivity"])
val_specificity_coim[4] <- as.numeric(results.glm.op_SE_N.coim$matrix.conf$byClass["Specificity"])
val_f1_coim[4] <- as.numeric(results.glm.op_SE_N.coim$matrix.conf$byClass["F1"])
val_recall_coim[4] <- as.numeric(results.glm.op_SE_N.coim$matrix.conf$byClass["Recall"])

# Guardar matriz de confusión del modelo (CSV, Latex)
write.csv(results.glm.op_SE_N.coim$matrix.conf$table, paste(PATH_RESULTS_GLM_COIM, "matriz_confianza_glm_variables_sin_colin_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(results.glm.op_SE_N.coim$matrix.conf$table, type = "latex"), file = paste(PATH_RESULTS_GLM_COIM, "matriz_confianza_glm_variables_sin_colin_con_normalizar.tex", sep=""))

# Guardar datos de predicciones
write.csv(results.glm.op_SE_N.coim$data.predicted, paste(PATH_RESULTS_GLM_COIM, "datos_prediccion_glm_variables_sin_colin_con_normalizar.csv", sep=""), row.names=TRUE)
print(xtable(as.data.frame(results.glm.op_SE_N.coim$data.predicted), type = "latex"), file = paste(PATH_RESULTS_GLM_COIM, "datos_prediccion_glm_variables_sin_colin_con_normalizar.tex", sep=""))

# Fusionar columnas de resultados
table_result_coim <- data.frame(cbind(val_tipo_coim, val_auc_coim, val_accuracy_coim, val_kappa_coim, val_sensitivity_coim, val_specificity_coim, val_f1_coim, val_recall_coim))
names(table_result_coim) <- c('Tipo', 'AUC', 'Accuracy', 'Kappa', 'Sensitivity', 'Specificity', 'F1', 'Recall')

# Guardar tabla de resultados
write.csv(table_result_coim, paste(PATH_RESULTS_GLM_COIM, "tabla_resultados_glm.csv", sep=""), row.names=TRUE)
print(xtable(table_result_coim, type = "latex"), file = paste(PATH_RESULTS_GLM_COIM, "tabla_resultados_glm.tex", sep=""))

# Finalizar la medición del tiempo de ejecución
hora.final <- Sys.time()

# Obtener tiempo de ejecución en horas
tiempo.ejec <- (hora.final - hora.inicio)[[1]]/3600

# Guardar valor del tiempo de ejecución
file.connection <- file("Datasets/Final/execution_time_code_logistic_regression.txt")
writeLines(c("Tiempo de ejecución",paste(tiempo.ejec,"horas",sep=" ")), file.connection)
close(file.connection)











































