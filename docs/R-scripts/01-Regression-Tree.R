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
data <- fread("./docs/R-scripts/Tectona.csv", stringsAsFactors=T)

## 4: Engenharia de recursos ----------------------------------------------
# data[,D2:=D^2
#      ][,lnD:=log(D)
#        ][,invD:=1/D
#          ][,D2H:=D^2*H
#            ][,lnH:=log(H)
#              ][,DH2:=D*H^2
#                ][,lnD2H:=log(D^2*H)
#                  ][,H2:=H^2
#                    ][,DH:=D*H]

## 5: Divisão aleatória estratificada ------------------------------------
set.seed(100)
trainIndex <- createDataPartition(y=data$V, p=.70, list=FALSE)
trainingSet <- data[trainIndex,]
testSet <- data[-trainIndex,]

## 6: Configuração do treinamento ----------------------------------------
source("./docs/R-scripts/Summary.R")

fitControl <- trainControl(method="cv", number=10,
                           returnResamp="final", savePredictions=TRUE,
                           allowParallel=T, summaryFunction=Summary,
                           verboseIter=T, selectionFunction="best")

## 7: Construção de modelos preditivos -----------------------------------

### 7.1: Hiperparâmetros candidatos --------------------------------------
tuneGrid <- expand.grid(cp = seq(0.0001, 0.02, 0.001))

tuneGrid <- expand.grid(kmax = seq(1,25,1),
                      kernel = c("biweight","epanechnikov",
                               "gaussian","triweight"),
                      distance = 1:3)

### 7.2: Ajuste de hiperparâmetros (Validação cruzada)
set.seed(1000)
m_RT <- train(V ~., data = trainingSet,
              method = "kknn",
              tuneGrid = tuneGrid,
              preProcess=c("center","scale", "BoxCox"),
              trControl = fitControl)

# Melhor configuração (ou valor de cp)
m_RT$bestTune

# Plot
plot(m_RT)

# imprime as decisões da árvore final
print(m_RT$finalModel)
rpart.rules(m_RT$finalModel, style = "wide")
rpart.rules(m_RT$finalModel, cover = TRUE)
(py <- partykit::as.party(m_RT$finalModel))

# Plota a árvore final
plot(m_RT$finalModel)
text(m_RT$finalModel)

rattle::fancyRpartPlot(m_RT$finalModel)
rpart.plot::rpart.plot(m_RT$finalModel)
plot(py)

# Desempenho médio na validação cruzada
results.m_RT <- m_RT$results
setorder(results.m_RT, RMSE)

# Variáveis importantes
varImp(m_RT)
ggplot(varImp(m_RT))

# Salva e ler os modelos
saveRDS(m_RT,'m_RT.rds')
# m_RT <- readRDS('m_RT.rds')

# Desempenho no conjunto de teste
(pred <- predict(m_RT, testSet))
df <- data.frame(obs = testSet$V, pred = pred)
round(Summary(data=df), 4)
