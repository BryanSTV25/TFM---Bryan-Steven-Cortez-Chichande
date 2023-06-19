## Cálculo de la huella de carbono al ejecutar código

# Desactivar salidas de números reales en notación científica
options(scipen=999)

# 1) Consumo de energía del equipo
# 95.35321364 W/h 

# 2) Factor de emisión de CO2 (gCO2/Wh) (Ecuador) - Agencia de Regulación y Control de Energía y Recursos Naturales no Renovables
# 0.1477 tonCO2/MWh

factor.ems.co2 <- (0.1477*1000)/1000000
factor.ems.co2

# 3) Tiempo de ejecución

# Inicio de la ejecución
hora.inicio <- Sys.time()

# Final de la ejecución
hora.final <- Sys.time()

# Obtener tiempo de ejecución en horas
tiempo.ejec <- (hora.final - hora.inicio)[[1]]/3600


# Huella de carbono = (Consumo de energía * Factor de emisión de CO2) * Tiempo de ejecución
# Gramos de CO2

val.huella.carb <- (95.35321364 * factor.ems.co2) * tiempo.ejec
val.huella.carb

# 0.000004621571 gCO2


(hora.final - hora.inicio)[[1]]

### Código final para calcular huella de carbono
val.consumo.pc <- 95.35321364 # Wh
tiempo.ejec <- 0.02801457 # h
factor.ems.co2 <- (0.1477*1000)/1000000 # gCO2/Wh
val.huella.carb <- (val.consumo.pc * factor.ems.co2) * tiempo.ejec
val.huella.carb
## Valor de huella de carbono: 0.0003945479 gCO2





