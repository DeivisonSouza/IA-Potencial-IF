###########################################################################
# Weighted k-Nearest Neighbors - wkNN
###########################################################################

## 1: Instala pacotes necessários -----------------------------------------
# install.packages(c("caret", "data.table", "tidyverse",
#                    "rpart", "rpart.plot", "easypackages"))

## 2: Carrega conjunto de dados
easypackages::libraries("caret","data.table","tidyverse",
                        "rpart", "rpart.plot")

## 3: Carrega conjunto de dados -------------------------------------------
data <- fread("./docs/R-scripts/Tectona.csv", stringsAsFactors=T)

## 4: Relação entre variáveis
ggplot(melt(data, id.vars=c("V")),
       aes(x=value, y=V)) + geom_point() +
  facet_grid( ~ variable)

## 5: Engenharia de recursos ----------------------------------------------
# data[,D2:=D^2
#      ][,lnD:=log(D)
#        ][,invD:=1/D
#          ][,D2H:=D^2*H
#            ][,lnH:=log(H)
#              ][,DH2:=D*H^2
#                ][,lnD2H:=log(D^2*H)
#                  ][,H2:=H^2
#                    ][,DH:=D*H]

## 6: Divisão aleatória estratificada ------------------------------------
set.seed(100)
trainIndex <- createDataPartition(y=data$V, p=.70, list=FALSE)
trainingSet <- data[trainIndex,]
testSet <- data[-trainIndex,]

## 7: Configuração do treinamento ----------------------------------------
source("./docs/R-scripts/Summary.R")

fitControl <- trainControl(method = "LOOCV",
                           returnResamp = "final",
                           savePredictions = TRUE,
                           allowParallel = T,
                           summaryFunction = Summary,
                           verboseIter = T,
                           selectionFunction = "best")

## 8: Construção de modelos preditivos -----------------------------------

### 8.1: Hiperparâmetros candidatos --------------------------------------
tuneGrid <- expand.grid(kmax = seq(1,10,1),
                        kernel = c("gaussian", "rectangular"),
                        distance = 2)

### 7.2: Ajuste de hiperparâmetros (Validação cruzada)
set.seed(1000)
m_knn <- train(V ~., data = trainingSet,
               method = "kknn",
               tuneGrid = tuneGrid,
               preProcess = c("center","scale", "BoxCox"),
               trControl = fitControl)

# Melhor configuração (ou valor de cp)
m_knn$bestTune

# Gráfico de desempenho usando LOOCV
plot(m_knn)

# Desempenho médio na validação cruzada
results.m_knn <- m_knn$results
setorder(results.m_knn, RMSE)

# Salva e ler os modelos
saveRDS(m_knn,'m_knn.rds')
# m_knn <- readRDS('m_knn.rds')

# Desempenho no conjunto de teste
(pred <- predict(m_knn, testSet))
df <- data.frame(obs = testSet$V, pred = pred)
round(Summary(data=df), 4)
