###########################################################################
# Regression Trees - RT
###########################################################################

## 1: Instala pacotes necessários -----------------------------------------
install.packages(c("caret", "data.table", "tidyverse",
                   "rpart", "rpart.plot", "easypackages"))

## 2: Carrega conjunto de dados
easypackages::libraries("caret","data.table","tidyverse",
                        "rpart", "rpart.plot")

## 3: Carrega conjunto de dados -------------------------------------------
data <- fread("Tectona.csv", stringsAsFactors=T)

## 4: Engenharia de recursos ----------------------------------------------
data[,lnVR:=log(VR)
][,D2:=D^2
][,lnD:=log(D)
][,invD:=1/D
][,D2H:=D^2*H
][,lnH:=log(H)
][,DH2:=D*H^2
][,lnD2H:=log(D^2*H)
][,H2:=H^2
][,DH:=D*H]

# Divisão aleatória estratificada
#----------------------------------------------------------
set.seed(100)
trainIndex <- createDataPartition(y=data$Esp,p=.80,list=FALSE)
trainingSet <- data[trainIndex,][,-"VR"]
testSet <- data[-trainIndex,][,-"VR"]

#----------------------------------------------------------
## CONFIGURACAO DO TREINAMENTO
#----------------------------------------------------------
source("Summary.R")

fitControl <- trainControl(method="cv",number=10,
                           returnResamp="final",savePredictions=TRUE,
                           allowParallel=T, summaryFunction=Summary,
                           verboseIter=T, selectionFunction="oneSE")

#----------------------------------------------------------
## CONSTRUÇÃO DE MODELOS PREDITIVOS
#----------------------------------------------------------
# Grade de hiperparâmetros
tuneGrid <- expand.grid(cp = seq(0.00001, 0.001, 0.00001))

# Ajuste de hiperparâmetros
set.seed(1000)
cartTuneD <- train(lnVR ~.,
                   data=trainingSet[,c(2,4:7)],
                   method="rpart",
                   tuneGrid=tuneGrid,
                   trControl=fitControl)

plot(cartTuneD)

# Melhor configuaração?
cartTuneD$bestTune

# Plot
plot(cartTuneD)

# imprime as decisões da árvore final
print(cartTuneD$finalModel)
rpart.rules(cartTuneD$finalModel, style = "wide")
rpart.rules(cartTuneD$finalModel, cover = TRUE)
(py<- partykit::as.party(cartTuneD$finalModel))

# Plota a árvore final
plot(cartTuneD$finalModel)
text(cartTuneD$finalModel)

rattle::fancyRpartPlot(cartTuneD$finalModel)
rpart.plot::rpart.plot(cartTuneD$finalModel)

plot(py)

source("treePlot.R")
treePlot(cartTuneD$finalModel, type = 2, extra = 101,
         varlen = 0, faclen = 1, fallen.leaves = TRUE)

# Medidas de desempenho médio?
results.cartTuneD <- cartTuneD$results
setorder(results.cartTuneD, RMSE)

(x <- results.cartTuneD[results.cartTuneD$cp == cartTuneD$bestTune$cp, ])

# Variáveis importantes
varImp(cartTuneD)
ggplot(varImp(cartTuneD))

# Salva e ler os modelos
saveRDS(cartTuneD,'Modelos/cartTuneD.rds')
#save(cartTuneD, file='Modelos/cartTuneDlnV.rda')
#cartTuneD<-readRDS('Modelos/cartTuneDlnV.rds')

# Desempenho no conjunto de treino
data <- data.frame(obs = trainingSet$lnVR,
                   pred = predict(cartTuneD, trainingSet))
round(Summary(data=data), 4)

# Desempenho no conjunto de teste
dataT <- data.frame(obs = testSet$lnVR, pred = predict(cartTuneD, testSet))
round(Summary(data=dataT), 4)
