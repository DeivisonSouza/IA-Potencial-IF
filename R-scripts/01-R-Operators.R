#####################################################################################                                                                                #
#     _____                                                                         #
#    | ___ \                                                                        #
#    | |_| | Curso: Linguagem de programação R: uma introdução para iniciantes      #
#    |    /  Universidade Federal do Pará (UFPA)                                    #
#    | |\ \  Facilitador: Deivison Venicio Souza                                    #
#    \_| \_|                                                                        #
#                                                                                   #
#####################################################################################

#------------------------------------------------------------------------------------
# OPERADORES EM R
#------------------------------------------------------------------------------------

# 1) Operadores aritméticos ---------------------------------------------------------
## Operadores aritméticos - Expressões básicas

2+3                    # adição
4*9                    # multiplicação
20/5                   # divisão
32-10                  # subtração
5^3                    # potenciação (exponenciação)
10%%3                  # resto da divisão
10%/%3                 # Parte inteira da divisão

(x <- c(2,4,6,8))
(y <- c(1,5,9,0))

x + y
x^y/x

## Operadores aritméticos - Expressões mais complexas

(2+3)*10
4*3**3
5+8-4*9/3
5+(8-4)*9/3
2^3*4+6/2
2^3*(4+6)/2

# 2) Operadores relacionais (ou comparação) --------------------------------------------

## criando vetores de um único elemento
(x <- 20)
(y <- 10)
(v <- 1)

## criando vetores com + de um elemento: "z" e "w"
(z <- c(1:5))
(w <- c(5:1))

x > y        # x é maior do que y?
x < y        # x é menor do que y?
x != y       # x é diferente de y?
y <= x       # y é menor ou igual a x?
y >= x       # y é maior ou igual a x?
z == w       # elementos de z são iguais aos de w?
z != w       # elementos de z são dif. aos de w?
x >= w       # x é >= aos elementos de w?
w %in% v     # w contém v?

## Um pouco mais sobre o operador `%in%`

(especie <- c("Vouacapoua-americana", "Cedrela-odorata", "Bertholletia-excelsa",
              "Dinizia-excelsa Ducke", "Bertholletia-excelsa", "Manilkara-huberi",
              "Couratari-guianensis"))

"Bertholletia-excelsa" %in% especie
"Swietenia-macrophylla" %in% especie

## Usando operador `%in%` para comparar dois vetores

(especie <- c("Vouacapoua-americana", "Cedrela-odorata", "Bertholletia-excelsa",
              "Dinizia-excelsa Ducke", "Bertholletia-excelsa", "Manilkara-huberi",
              "Couratari-guianensis"))

(ameacadas <- c("Vouacapoua-americana", "Bertholletia-excelsa", "Cedrela-odorata",
                "Swietenia-macrophylla"))

especie %in% ameacadas

# 3) Operadores lógicos ----------------------------------------------------------------
library(readr)
(data <- read_csv("Data/data1.csv"))

# Quais árvores possuem DAP maior ou igual a 50cm e qualidade de fuste (QF) igual a 1?

data$DAP >= 50 & data$QF == 1

# Quais árvores possuem cipós ausentes, QF igual a 1 e DAP maior ou igual a 50cm?

data$Cipo == "Não" & data$QF == 1 & data$DAP >= 50

# Quais árvores estão selecionadas para explorar e QF igual a 1?

data$Selecao == "Explorar" & data$QF == 1
