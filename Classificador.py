import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report#, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
#nltk.download('punkt')
import string
# nlp = spacy.load('pt')

classificadorgp1 = AdaBoostClassifier(n_estimators=200)
classificadorgp2 = 0
classificadorgp3 = AdaBoostClassifier(n_estimators=200)
classificadorgp4 = 0
classificadorgp5 = AdaBoostClassifier(n_estimators=200)
classificadorgp6 = AdaBoostClassifier(n_estimators=200)
classificadorgp7 = 0
classificadorft = AdaBoostClassifier(n_estimators=200)
classificadorfp = AdaBoostClassifier(n_estimators=200)
classificadorfr = 0
classificadorfs = AdaBoostClassifier(n_estimators=200)

def carregar_dados(csv):

    data = pd.read_csv(csv)

    classe = data.keys()[-1] #Pega o nome da coluna das classes

    y = data[classe] #y Recebe as classes de todas as instâncias da base de dados

    for i in data:
        i = str(i)
    X = data.copy()
    del X[classe] #Recebe os demais atributos

    return data, X.values.tolist(), y.tolist()

def erro_por_classe(matriz_confusao, resultados):
    # A PARTIR DA MATRIZ DE CONFUSÃO CALCULA O ERRO POR CLASSE
    # **
    # matriz_confusao - soma das matrize de confusão de cada fold do tipo ndarray(numpy) de forma NxN
    # resultados - dicionário de todos os resultados
    # **
    # retorna o dicionário resultados com os erros de cada classe preenchidos

    tam = matriz_confusao.shape[0]

    for i in range(tam):

        acerto = matriz_confusao[i][i]
        total = sum(matriz_confusao[i])

        taxa_erro = round(1 - (acerto / total),4)

        resultados["erro_classe_"+str(i)].append(taxa_erro)


def validacao_cruzada(X, y, features, k, ntree, resultados ):
    ##É REALIZADO OS O EXPERIMENTO COM VALIDAÇÃO CRUZADA E OS RESULTADOS É ADICIONADO A UM DICIONÁRIO
    # **
    # X - dados
    # y - classes
    # k - número de folds
    # ntree - Número de árvores
    # mtry - número de features
    # metricas - lista de metricas que serão utilizadas na avaliacão( "acurácia","kappa", "OOB_erro")
    # resultados - dicionário que vai ser utilizado para cada experimento, salvando os resultados em um dicionário para ser salvo em CSV
    # **
    # retorna o dicionário resultados com os resultados desse experimento adicionados

    resultados_parciais = {} #SALVAR RESULTADOS DE CADA RODADA DA VALIDAÇÃO CRUZADA
    resultados_parciais.update({'ntree': []})
    resultados_parciais.update({'mtry': []})
    resultados_parciais.update({'acurácia': []})
    resultados_parciais.update({'kappa': []})
    resultados_parciais.update({'accuracy': []})
    resultados_parciais.update({'erro': []})

    ## VALIDAÇÃO CRUZADA

    rkf = RepeatedStratifiedKFold(n_splits=k, n_repeats=1, random_state=54321) #DIVIDI OS DADOS NOS CONJUNTOS QUE SERÃO DE      TREINO E TESTE EM CADA RODADA DA VALIDAÇÃO CRUZZADA

    matriz_confusao = np.zeros((2,2))


    for train_index, test_index in rkf.split(X, y):
        X_train, X_test = [X.iloc[i] for i in train_index], [X.iloc[j] for j in test_index]
        y_train, y_test = [y.iloc[i] for i in train_index], [y.iloc[j] for j in test_index]

        X_train_np = np.asarray(X_train)
        X_test_np = np.asarray(X_test)
        y_train_np = np.asarray(y_train)
        y_test_np = np.asarray(y_test)

        classificador = AdaBoostClassifier(n_estimators=ntree)
        classificador.fit(X_train_np, y_train_np)
        y_pred = classificador.predict(X_test_np)

        resultados_parciais["acurácia"].append(accuracy_score(y_pred, y_test_np))
        resultados_parciais["kappa"].append(cohen_kappa_score(y_pred, y_test_np))

        matriz_confusao = matriz_confusao + confusion_matrix(y_pred=y_pred, y_true=y_test_np) ##A MATRIZ DE CONFUSÃO FINAL SERÁ A SOMA DAS MATRIZES DE CONFUSÃO DE CADA RODADA DO KFOLD


    ## SALVANDO OS PARÊMTROS E RESULTADOS DO EXPERIMENTO


    #print(matriz_confusao)
    resultados['ntree'].append(classificador.n_estimators)
    erro_por_classe(matriz_confusao, resultados)

    media = np.mean(resultados_parciais["acurácia"])
    std = np.std(resultados_parciais["acurácia"])
    resultados["acurácia"].append(str(round(media,4))+"("+str(round(std,4))+")")

    resultados["accuracy"].append(round(media, 4))
    resultados["erro"].append(round(1 - media, 4))

    media = np.mean(resultados_parciais["kappa"])
    std = np.std(resultados_parciais["kappa"])
    resultados["kappa"].append(str(round(media, 4)) + "(" + str(round(std, 4)) + ")")



    return resultados, classificador



def experimentos(banco):

##CARREGAR OS DADOS


    dataset = pd.read_csv(banco)
    features = dataset.columns.difference(['id_feedback','class'])

    print(features)

    X = dataset.iloc[:, 1:-1]
    y = dataset.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    print(X_train)
    print(X_test)
    print(y_train)
    print(y_test)

    resultados = {}
    resultados.update({'ntree': []})
    resultados.update({'mtry': []})
    resultados.update({'acurácia': []})
    resultados.update({'kappa': []})
    resultados.update({'accuracy': []})
    resultados.update({'erro': []})


    classes = set(y_train)
    for i in classes:
        resultados.update({"erro_classe_"+str(i):[]})

    j = 0
    maior = 0.0
    best_mtree = 0

    resultados, classificador = validacao_cruzada(X_train, y_train, features, k=10, ntree=200, resultados=resultados)

    print(resultados)

    return classificador


def search(lista, valor):
    return [lista.index(x) for x in lista if valor in x]

def extractLiwc(text):

    # reading liwc
    wn = open('LIWC2007_Portugues_win.dic.txt', 'r', encoding='ansi', errors='ignore').read().split('\n')
    wordSetLiwc = []
    for line in wn:
        words = line.split('\t')
        if (words != []):
            wordSetLiwc.append(words)

    # indexes of liwc
    indices = open('indices.txt', 'r', encoding='utf-8', errors='ignore').read().split('\n')

    # dataset tokenization
    wordsDataSet = []

    wordsLine = []
    for word in word_tokenize(text):
        if word not in string.punctuation + "\..." and word != '``' and word != '"':
            wordsLine.append(word.lower())


    # initializing liwc with zero
    liwc = [0] * len(indices)
    #liwc.append([0] * len(indices))

    # performing couting

    print("writing liwc ")
    print(liwc)

    for word in wordsLine:
        position = search(wordSetLiwc, word)
        if position != []:
            tam = len(wordSetLiwc[position[0]])
            for i in range(tam):
                if wordSetLiwc[position[0]][i] in indices:
                    positionIndices = search(indices, wordSetLiwc[position[0]][i])
                    liwc[positionIndices[0]] = liwc[positionIndices[0]] + 1

    return liwc
#------------------------------------ADITIONAL FEATURES---------------------------------------------------------------#
#aditional features
def aditionals(post):
    postOriginal = post.lower()
    #post = nlp(post)

    greeting = sum([word_tokenize(postOriginal).count(word) for word in ['olá', 'oi', 'como vai', 'tudo bem', 'como está', 'como esta', 'bom dia', 'boa tarde', 'boa noite']])
    compliment = sum([word_tokenize(postOriginal).count(word) for word in ['parabéns', 'parabens', 'excelente', 'fantástico', 'fantastico', 'bom', 'bem', 'muito bom', 'muito bem', 'ótimo', 'otimo', 'incrivel', 'incrível', 'maravilhoso', 'sensacional','irrepreensível', 'irrepreensivel', 'perfeito']])
    #ners = len(post.ents)

    return [greeting, compliment]


def processa(texto):

    data = {}
    liwc = []
    liwc = extractLiwc(texto)
    print("liwc = ",liwc)
    adds = aditionals(texto)
    cohmetrix = [50.0, 0.0, 500.0, 86.405, 450.0, 2.0, 2.5, 10.0, 150.0, 1.0, 2.0, 20.0, 100.0, 300.0, 50.0, 0.0, 0.0,
                 0.0, 50.0, 76562.5, 3441.5, 1.0, 0.0, 0.0, 0.95, 0.25, 250.0, 0.0, 150.0, 0.0, 50.0, 0.0, 100.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 1.66666666666667, 10.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    features = []
    for x in liwc:
        features.append(x)
    for y in cohmetrix:
        features.append(y)
    features.append(adds[0])
    features.append(adds[1])
    features.append(0)
    features.append(0)

    newfeatures = []

    newfeatures.append(features)

    global classificadorgp1
    global classificadorgp2
    global classificadorgp3
    global classificadorgp4
    global classificadorgp5
    global classificadorgp6
    global classificadorgp7
    global classificadorft
    global classificadorfp
    global classificadorfs
    global classificadorfr

    classes = []

    y_pred_gp1 = classificadorgp1.predict(newfeatures)
    y_pred_gp2 = 0
    y_pred_gp3 = classificadorgp3.predict(newfeatures)
    y_pred_gp4 = 0
    y_pred_gp5 = classificadorgp5.predict(newfeatures)
    y_pred_gp6 = classificadorgp6.predict(newfeatures)
    y_pred_gp7 = 0
    y_pred_ft = classificadorft.predict(newfeatures)
    y_pred_fp = classificadorfp.predict(newfeatures)
    y_pred_fr = 0
    y_pred_fs = classificadorfs.predict(newfeatures)

    classes.append(y_pred_gp1[0])
    classes.append(y_pred_gp2)
    classes.append(y_pred_gp3[0])
    classes.append(y_pred_gp4)
    classes.append(y_pred_gp5[0])
    classes.append(y_pred_gp6[0])
    classes.append(y_pred_gp7)
    classes.append(y_pred_ft[0])
    classes.append(y_pred_fp[0])
    classes.append(y_pred_fr)
    classes.append(y_pred_fs[0])

    return classes

def carregaClassificadores():

    bancogp1 = "banco-gp1.csv"
    bancogp3 = "banco-gp3.csv"
    bancogp5 = "banco-gp5.csv"
    bancogp6 = "banco-gp6.csv"
    bancoft = "banco-lak-FT.csv"
    bancofp = "banco-lak-FP.csv"
    bancofs = "banco-lak-FS.csv"

    global classificadorgp1
    global classificadorgp2
    global classificadorgp3
    global classificadorgp4
    global classificadorgp5
    global classificadorgp6
    global classificadorgp7
    global classificadorft
    global classificadorfp
    global classificadorfs
    global classificadorfr

    classificadorgp1 = experimentos(bancogp1)
    classificadorgp2 = 0
    classificadorgp3 = experimentos(bancogp3)
    classificadorgp4 = 0
    classificadorgp5 = experimentos(bancogp5)
    classificadorgp6 = experimentos(bancogp6)
    classificadorgp7 = 0
    classificadorft = experimentos(bancoft)
    classificadorfp = experimentos(bancofp)
    classificadorfr = 0
    classificadorfs = experimentos(bancofs)


#teste
carregaClassificadores()
texto = "Boa noite Muito bom Parabéns"
vetor = processa(texto)

for elemento in vetor:
    print(elemento)

