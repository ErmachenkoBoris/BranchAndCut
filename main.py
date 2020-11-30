import multiprocessing
import urllib.request
import time
import cplex
import random
import math
import numpy as np
import copy

# GLOBAL
timeLimitValue = 3600 * 3
delta = 0.00001
local = True
bestDecision = 0
maxColorGlobal = []
grath = {}
coloredEdValue = []

paths = [
    'http://iridia.ulb.ac.be/~fmascia/files/DIMACS/C125.9.clq',
    'http://iridia.ulb.ac.be/~fmascia/files/DIMACS/brock200_2.clq',
    'http://iridia.ulb.ac.be/~fmascia/files/DIMACS/keller4.clq'
]

localPaths = [
    # 'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/c-fat200-1.clq',
    # 'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/c-fat200-2.clq',
    # 'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/c-fat200-5.clq',
    # 'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/c-fat500-1.clq',
    # 'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/c-fat500-10.clq',
    # 'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/c-fat500-2.clq',
    # 'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/c-fat500-5.clq',
    # 'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/MANN_a9.clq',
    # 'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/hamming6-2.clq',
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/hamming6-4.clq',
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/gen200_p0.9_44.clq',
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/gen200_p0.9_55.clq',
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/san200_0.7_1.clq',
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/san200_0.9_1.clq',
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/san200_0.9_2.clq',
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/san200_0.9_3.clq',
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/sanr200_0.7.clq',
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/C125.9.clq',
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/keller4.clq',
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/brock200_1.clq',
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/brock200_2.clq',
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/brock200_3.clq',
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/brock200_4.clq',
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/p_hat300-1.clq',
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/p_hat300-2.clq'
]

localHARDPaths = [
    # 'graphs/san200_0.7_2.clq'
    # 'graphs/gen200_p0.9_44.clq',
    # 'graphs/gen200_p0.9_55.clq',
    # 'graphs/san200_0.7_1.clq',
    # 'graphs/san200_0.9_1.clq',
    # 'graphs/san200_0.9_2.clq',
    'graphs/C125.9.clq',
    'graphs/san200_0.9_3.clq',
    # 'graphs/sanr200_0.7.clq',

    # 'graphs/keller4.clq',
    # 'graphs/brock200_1.clq',
    # 'graphs/brock200_3.clq',
    # 'graphs/brock200_4.clq',
    # 'graphs/p_hat300-2.clq'
]

# -----------------------------------------------TEST-------------------------------
# matrixTest = np.zeros((4, 4))
# matrixTest = np.array([
#     [1, 1, 0, 0],
#     [1, 1, 1, 0],
#     [0, 1, 1, 1],
#     [1, 0, 1, 1]])

# matrixTest = np.array([
#     [1, 1, 0, 0, 0],
#     [1, 1, 1, 1, 1],
#     [0, 1, 1, 0, 0],
#     [0, 1, 0, 1, 0],
#     [0, 1, 0, 0, 1]])

matrixTest = np.array([
    [1, 1, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 1, 1]])


# matrixTest = np.array([
#             [1,1,1,1],
#            [1,1,1,1],
#           [1,1,1,1],
#     [1,1,1,1]])
# -----------------------------------------------TEST-------------------------------


# --------------------OPEN FILE--------------------
def openGraph(filePath):
    n = -1
    m = -1
    global local
    if local == True:
        file = open(filePath)
    else:
        file = urllib.request.urlopen(filePath)
    for line in file:
        if local == False:
            line = line.decode('ascii')
            line = line.strip('\n')
        if line.startswith('p'):
            n = int(line.split(' ')[2])
            m = int(line.split(' ')[3])
            break
    graphMatrix = np.zeros((n, n))
    for line in file:
        if local == False:
            line = line.decode('ascii')
            line = line.strip('\n')
        if line.startswith('e'):
            i = int(line.split(' ')[1]) - 1
            j = int(line.split(' ')[2]) - 1
            graphMatrix[i, j] = 1
            graphMatrix[j, i] = 1
    return n, m, graphMatrix


def graphByNeighborhoods(graphMatrix, n):
    graphModel = [[] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and graphMatrix[i][j] == 1:
                graphModel[i].append(j)
    return graphModel


# ---------------------------------------------------------------------
# ----------------------Heuristic Functions--------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# красим жадно
def colorGreedy(matrix, edges):
    V = [i for i in range(edges)]
    colorGroups = [[]]
    coloredV = [-1 for p in range(edges)]
    k = 0
    for i in range(edges):
        if i not in V:
            continue
        colorGroups[k].append(i)
        V.remove(i)
        while len(matrix[i].nonzero()[0]) != edges:  # пока есть ненулевые
            for j in range(i, edges):  # ?
                if matrix[i, j] == 0 and j in V:
                    break
            if j == edges:
                break
            if j == edges - 1 and matrix[i, j] != 0 or j not in V:
                break

            colorGroups[k].append(j)
            V.remove(j)
            matrix[i] = matrix[i] + matrix[j]

        k = k + 1
        colorGroups.insert(k, [])
    for i in range(k):
        for j in range(len(colorGroups[i])):
            coloredV[colorGroups[i][j]] = i
    return coloredV


# находим узлы, которые связаны с бОльшим числом разноцветных соседей:
# (начнем евристику с них)
def getWithMaxColorNumber(matrix, n, coloredEdges):
    maxColorCount = 0
    maxColorCountEdges = []
    for i in range(n):
        colorTmpCounter = []
        for j in range(n):
            if matrix[i, j] == 1 and i != j and coloredEdges[j] not in colorTmpCounter:
                colorTmpCounter.append(coloredEdges[j])
        if len(colorTmpCounter) == maxColorCount:
            maxColorCountEdges.append(i)
        if len(colorTmpCounter) > maxColorCount:
            maxColorCount = len(colorTmpCounter)
            maxColorCountEdges = [i]
    return maxColorCountEdges


# эвристический поиск клики (с помощью раскрашенного графа)
def findEvristicClique(maxColorCandidatARRAY, matrix, n, coloredEd):

    # # Находим соседей выбранных вершин и их пересечение в следующих циклах
    # lengthData = len(maxColorCandidatARRAY)
    # listOfNeiborsList = [[] for _ in range(0, lengthData)]
    # for j in range(len(maxColorCandidatARRAY)):
    #     for i in range(n):
    #         if matrix[maxColorCandidatARRAY[j], i] == 1 and i != maxColorCandidatARRAY[j]:
    #             listOfNeiborsList[j].append(i)
    #
    # clickCandidat = listOfNeiborsList[0]
    #
    # for k in range(lengthData):
    #     clickCandidat = list(set(clickCandidat) & set(listOfNeiborsList[k]))
    #
    # if (len(clickCandidat) > 0):
    #     clickEvr = maxColorCandidatARRAY
    # else:
    #     clickEvr = [random.choice(maxColorCandidatARRAY)]
    #     for i in range(n):
    #         if matrix[clickEvr[0], i] == 1 and i != clickEvr[0]:
    #             clickCandidat.append(i)
    #
    # def findClickEvr(clickEvrF, clickCandidatF, matrix):
    #     clickCandidatFCLEAR = []
    #     for z in clickCandidatF:
    #         if z not in clickEvrF:
    #             clickCandidatFCLEAR.append(z)
    #
    #     maxColorLocal = random.choice(clickCandidatFCLEAR)
    #
    #     clickEvrF.append(maxColorLocal)  # добавили в клику
    #
    #     clickLocalCandidat = []  # ищем соседей новых
    #     for i in range(n):
    #         if matrix[maxColorLocal, i] == 1 and i != maxColorLocal:
    #             clickLocalCandidat.append(i)
    #
    #     # находим пересечение со старыми соседями
    #     newCandidats = list(set(clickCandidatF) & set(clickLocalCandidat))
    #     if len(newCandidats) > 0:
    #         return findClickEvr(clickEvrF, newCandidats, matrix)
    #     return clickEvrF
    #
    # return findClickEvr(clickEvr, clickCandidat, matrix)
    # print('erfsdfsdf ', maxColorCandidatARRAY)

    # Находим соседей выбранных вершин и их пересечение в следующих циклах
    lengthData = len(maxColorCandidatARRAY)
    listOfNeiborsList = [[] for _ in range(0, lengthData)]
    for j in range(len(maxColorCandidatARRAY)):
        for i in range(n):
            if matrix[maxColorCandidatARRAY[j], i] == 1 and i != maxColorCandidatARRAY[j]:
                listOfNeiborsList[j].append(i)

    clickEvr = []
    clickCandidat = listOfNeiborsList[0]

    # print('listOfNeiborsList ', listOfNeiborsList)
    for k in range(lengthData):
        clickCandidat = list(set(clickCandidat) & set(listOfNeiborsList[k]))

    if (len(clickCandidat) > 0):
        clickEvr = maxColorCandidatARRAY
        # print(' !!!!-- > ',len(clickEvr))
    else:
        # print('gabella')
        clickEvr = [random.choice(maxColorCandidatARRAY)]
        for i in range(n):
            if matrix[clickEvr[0], i] == 1 and i != clickEvr[0]:
                clickCandidat.append(i)

    def findClickEvr(clickEvrF, clickCandidatF, matrix):
        maxColorLocalValue = -1
        maxColorLocal = clickCandidatF[0]

        # ищем максимального соседа
        # for z in clickCandidatF:
        #     if coloredEd[z] >=maxColorLocalValue and z not in clickEvrF:
        #         maxColorLocalValue = coloredEd[z]
        #         maxColorLocal.append(z)

        # maxColorLocalArray = []
        # for z in clickCandidatF:
        #     if coloredEd[z] >=maxColorLocalValue and z not in clickEvrF:
        #         if coloredEd[z] > maxColorLocalValue:
        #             maxColorLocal=[]
        #         maxColorLocalValue = coloredEd[z]
        #         maxColorLocalArray.append(z)

        clickCandidatFCLEAR = []
        for z in clickCandidatF:
            if z not in clickEvrF:
                clickCandidatFCLEAR.append(z)

        maxColorLocal = random.choice(clickCandidatFCLEAR)

        clickEvrF.append(maxColorLocal)  # добавили в клику

        clickLocalCandidat = []  # ищем соседей новых
        for i in range(n):
            if matrix[maxColorLocal, i] == 1 and i != maxColorLocal:
                clickLocalCandidat.append(i)

        # находим пересечение со старыми соседями
        newCandidats = list(set(clickCandidatF) & set(clickLocalCandidat))
        if len(newCandidats) > 0:
            return findClickEvr(clickEvrF, newCandidats, matrix)
        return clickEvrF

    return findClickEvr(clickEvr, clickCandidat, matrix)


# запуск всего механизма эвристики
def evristic(matrix, n, path):

    # start_evr_time = time.time()
    #
    # print('---evristic')
    # n, m, matrix = openGraph(path)
    #
    # coloredEdValue = colorGreedy(matrix.copy(), n)
    # maxColor = getWithMaxColorNumber(matrix.copy(), n, copy.copy(coloredEdValue))
    #
    # bestEvrValue = -1
    # bestEvrStore = []
    #
    # for i in range(5000):
    #     randomEdges = random.sample(maxColor, 3)
    #     clickEvristic = findEvristicClique(randomEdges, matrix.copy(), n, copy.copy(coloredEdValue))
    #     if len(clickEvristic) >= bestEvrValue:
    #         if len(clickEvristic) > bestEvrValue:
    #             bestEvrStore = []
    #         bestEvrValue = len(clickEvristic)
    #         bestEvrStore.append(clickEvristic)
    #
    # bestEvrFinal = random.choice(bestEvrStore)
    # bestEvrValue = len(bestEvrFinal)
    #
    # clickValue = [0 for i in range(n)]
    # for i in bestEvrFinal:
    #     clickValue[i] = 1
    #
    # print('--- grath Name: ', path)  # Название графа
    # print("--- seconds Heuristics: %s" % (time.time() - start_evr_time))  # время эвристики
    # print('--- colors of vertexes: ', coloredEdValue)  # раскраска графа
    # print('--- maxColor connected vertex:', maxColor)  # узлы с наибольшим количеством разноцветных соседей
    # print('--- Heuristics values:', bestEvrFinal)  # решение клики
    # print('--- Heuristics Power', bestEvrValue)  # количество узлов в клике
    # print('')
    print('---evristic')
    n, m, confusion_matrix = openGraph(path)

    # ----------------TEST
    # n = 4
    # confusion_matrix = matrixTest
    # ----------------TEST

    start_evr_time = time.time()

    coloredEd = colorGreedy(confusion_matrix.copy(), n)

    maxColor = getWithMaxColorNumber(confusion_matrix.copy(), n, coloredEd)

    bestEvrValue = -1
    bestEvrFinal = []
    bestEvrStore = []

    print('--------------maxColor---------- ', len(maxColor))

    for i in range(5000):
        randomEdges = random.sample(maxColor, 3)
        clickEvristic = findEvristicClique(randomEdges, confusion_matrix.copy(), n, coloredEd)
        if len(clickEvristic) >= bestEvrValue:
            if len(clickEvristic) > bestEvrValue:
                bestEvrStore = []
            bestEvrValue = len(clickEvristic)
            bestEvrStore.append(clickEvristic)

    bestEvrFinal = random.choice(bestEvrStore)
    bestEvrValue = len(bestEvrFinal)

    clickValue = [0 for i in range(n)]
    for i in bestEvrFinal:
        clickValue[i] = 1


    print('--------------- grath N ', path) # Название графа
    print("--- %s seconds EVRISTIC---" % (time.time() - start_evr_time)) # время эвристики
    print('coloredEd ', coloredEd) # раскраска графа
    print('maxColor ', maxColor) # узлы с наибольшим количеством разноцветных соседей
    print('bestEvr ', bestEvrFinal) # решение клики
    print('clickEvristicPower ', bestEvrValue) # количество узлов в клике
    print(clickValue) # решение клики в 0,1
    print('--------------- grath N ', path) # Название графа
    print('')
    print('')
    coloredEdValue = coloredEd
    return bestEvrValue, clickValue, matrix, n, coloredEdValue


# ---------------------------------------------------------------------
def indSetSearch(neighborsGraph, weight, coloredEd):
    indSet = evristicIndSetSearch(coloredEd)
    sumMax = 0
    indSetMax = []
    for i in range(100):
        sum, indSet = localSearch(copy.copy(neighborsGraph), copy.copy(indSet), copy.copy(weight))
        if sumMax < sum:
            sumMax = sum
            indSetMax = copy.copy(indSet)
    return sumMax, indSetMax

def evristicIndSetSearch(colored):
    randomVertex = random.randint(0, len(colored)-1)
    randomColor = colored[randomVertex]
    indSet = []
    for i in range(len(colored)):
        if colored[i] == randomColor:
            indSet.append(i)
    return indSet

# локальный поиск
def localSearch(graphNeighbors, indSet, weight):
    N = len(graphNeighbors)
    # statusArray:
    # 1 - indSet
    # 2 - freeVertex
    # 3 - bindedVertex
    tightness = [0 for i in range(N)]
    statusArray = [2 for i in range(N)]
    for i in range(N):
        if i in indSet:
            statusArray[i] = 1
            for j in range(len(graphNeighbors[i])):
                tightness[graphNeighbors[i][j]] += 1
                statusArray[graphNeighbors[i][j]] = 3
        else:
            if tightness[i] == 0:
                statusArray[i] = 2
    candidatsVertex = []

    for i in range(len(indSet)):
        tightCount = 0
        for j in range(len(graphNeighbors[i])):
            if tightness[graphNeighbors[i][j]] == 1:
                tightCount+=1
                if tightCount >= 2:
                    candidatsVertex.append(i)
                    break

    # print('1-')
    while len(candidatsVertex) > 0 :
        # print('hii')
        vertForSwap = random.choice(candidatsVertex)
        u = -1
        v = -1
        for i in range(len(graphNeighbors[vertForSwap])):
            if (tightness[graphNeighbors[vertForSwap][i]] == 1):
                if u == -1:
                    u = graphNeighbors[vertForSwap][i]
                else:
                    if graphNeighbors[vertForSwap][i] not in graphNeighbors[u]:
                        v = graphNeighbors[vertForSwap][i]
            if u != -1 and v != -1:
                break
        if u != -1 and v != -1:
            # print('123')

            statusArray[u] = statusArray[v] = 1
            statusArray[vertForSwap] = 3
            # ???
            # candidatsVertex.append(u)
            # candidatsVertex.append(v)
            candidatsVertex.remove(vertForSwap)
            for j in range(len(graphNeighbors[vertForSwap])):
                tmpIndex = graphNeighbors[vertForSwap][j]
                tightness[tmpIndex] -= 1
                # ???
                # if tightness[tmpIndex] == 0 and statusArray[tmpIndex] == 3:
                #     statusArray[tmpIndex] = 2
                # if tightness[tmpIndex] == 1:
                #     for z in range(len(statusArray)):
                #         if statusArray[z] == 1:
                #             if tmpIndex in graphNeighbors[statusArray[z]]:
                #                 candidatsVertex.append(z)
            #
            for j in range(len(graphNeighbors[u])):
                tmpIndex = graphNeighbors[u][j]
                tightness[tmpIndex] += 1


                # if tightness[tmpIndex] == 1:
                #     statusArray[tmpIndex] = 3

            for j in range(len(graphNeighbors[v])):
                tmpIndex = graphNeighbors[v][j]
                tightness[tmpIndex] += 1
                # if tightness[tmpIndex] == 1:
                #     statusArray[tmpIndex] = 3
            for j in range(len(tightness)):
                if tightness[j] == 0 and statusArray[j] == 3:
                    statusArray[j] = 2
                if tightness[j] == 1:
                    countT = 0
                    for z in range(len(statusArray)):
                        if statusArray[z] == 1:
                            if j in graphNeighbors[z] and z not in candidatsVertex:
                                countT = countT + 1
                                if countT >= 2:
                                    candidatsVertex.append(z)
                                    # print('0---0')

                                    break
        else:
            # print('sed')
            candidatsVertex.remove(vertForSwap)
            continue
    # print('2-')

    sumWeight = 0
    answer = []
    for i in range(len(statusArray)):
        if statusArray[i] == 1:
            sumWeight += weight[i]
            answer.append(i)
    return sumWeight, answer

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

# Инициализируем модель cplex (добавляем все ограничения)
def initalClickCPLEX(constrains,
                     constrainsNames,
                     constrainsTypes,
                     constrainsRightParts,
                     maxCliqueModel,
                     matrix,
                     n):
    maxCliqueModel.variables.add(names=["y" + str(i) for i in range(n)],
                                 types=[maxCliqueModel.variables.type.continuous for i in range(n)])

    for i in range(n):
        maxCliqueModel.variables.set_lower_bounds(i, 0.0)
        maxCliqueModel.variables.set_upper_bounds(i, 1.0)

    # for i in range(matrix.shape[0]):
    #     for j in range(i + 1, matrix.shape[1]):
    #         if matrix[i][j] == 0.0:
    #             constrains.append([["y" + str(i), "y" + str(j)], [1, 1]])
    #             constrainsNames.append("constraint_" + str(i) + "_" + str(j))
    #             constrainsTypes.append('L')
    #             constrainsRightParts.append(1.0)
    #
    # maxCliqueModel.linear_constraints.add(
    #     lin_expr=constrains,
    #     rhs=constrainsRightParts,
    #     names=constrainsNames,
    #     senses=constrainsTypes)

    maxCliqueModel.set_log_stream(None)
    maxCliqueModel.set_warning_stream(None)
    maxCliqueModel.set_results_stream(None)

    for i in range(n):
        maxCliqueModel.objective.set_linear("y" + str(i), 1)

    maxCliqueModel.objective.set_sense(maxCliqueModel.objective.sense.maximize)

    maxCliqueModel.solve()
    values = maxCliqueModel.solution.get_values()
    result = 0
    for v in values:
        result = result + v
    return values


# ------------------------BNC----------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
# функция, которую запускаем в отедльном процессе, чтобы была возможность остановить по времени
def bncContainer(evristicPower, evristicValues, matrix, n, graphName, return_dict, coloredEd, grathNeighborsV):
    start_BNC_time = time.time()
    constrains = []
    constrainsNames = []
    constrainsTypes = []
    constrainsRightParts = []

    global bestDecision
    global coloredEdValue
    global grath

    coloredEdValue = coloredEd
    grath = grathNeighborsV

    maxCliqueModel = cplex.Cplex()

    initalClickCPLEX(
        constrains,
        constrainsNames,
        constrainsTypes,
        constrainsRightParts,
        maxCliqueModel,
        matrix,
        n, )

    bestDecision = evristicPower

    return_dict['power'] = evristicPower
    return_dict['values'] = evristicValues

    result, resultValues = BNC(evristicValues, maxCliqueModel, return_dict)
    print("--- seconds BNC: %s " % (time.time() - start_BNC_time))
    print('--- !! result Power: ', result)
    print('--- !! resultValues: ', resultValues)
    print('')
    print('')
    print('')
    print('')

#     проход по всем графам из файлов и запуск эвристики и bnc для каждого
def bncStartEngine(graphs):
    for i in range(len(graphs)):

        global grath
        global coloredEdValue
        n, m, confusion_matrix = openGraph(graphs[i])
        grath = graphByNeighborhoods(confusion_matrix, len(confusion_matrix))
        evristicPower, evristicValues, matrix, n, coloredEdValue = evristic(confusion_matrix, n, graphs[i])
        if __name__ == '__main__':
            global timeLimitValue
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            return_dict['power'] = 0
            return_dict['values'] = []
            print('start bnC for ', graphs[i])
            p = multiprocessing.Process(target=bncContainer,
                                        args=(evristicPower, evristicValues, matrix, n, graphs[i], return_dict, coloredEdValue, grath))
            p.start()
            p.join(timeLimitValue)
            if p.is_alive():
                p.terminate()
                print("--- %s seconds LIMIT BNC: " % timeLimitValue)
                print('!!!!! TIMEOUT result: ', return_dict['power'])
                print('!!!!! TIMEOUT resultValues: ', return_dict['values'])


# округление с учетом дельты
def numberWithDelta(number, maxValue, minVale, eps):
    if number + eps >= maxValue:
        return maxValue
    if number - eps <= minVale:
        return minVale
    return number


# добавление ограничения
def addConstrain(i, value, maxCliqueModel):
    maxCliqueModel.linear_constraints.add(
        lin_expr=[[["y" + str(i)], [1]]],
        rhs=[value],
        names=["constraint_" + str(i)],
        senses=['E']
    )


# удаление ограничения
def removeConstrain(i, maxCliqueModel, full):
    if full == True:
        maxCliqueModel.linear_constraints.delete(i)
    else:
        maxCliqueModel.linear_constraints.delete("constraint_" + str(i))


# решение сиплексом
def solveWithCPLX(maxCliqueModel):
    maxCliqueModel.solve()
    return maxCliqueModel.solution.get_values()

def branching(currentDecisionValue):
    flag = False
    for index in range(len(currentDecisionValue)):
        currentDecisionValue[index] = numberWithDelta(currentDecisionValue[index], 1, 0, delta)
        if currentDecisionValue[index] != 0 and currentDecisionValue[index] != 1:
            flag = True
            return index, flag
    return index, flag

def addComplexConstrain(maxCliqueModel, indSetMax):

    constrains = []
    constrainsNames = []
    constrainsTypes = []
    constrainsRightParts = []

    variables = []
    constrainsName = "constraint"

    for i in range(len(indSetMax)):
        variables.append("y" + str(indSetMax[i]))
        constrainsName = constrainsName + "_" + str(indSetMax[i])
    coef = [1] * len(indSetMax)
    constrains.append([variables])
    constrainsNames.append(constrainsName)
    constrainsTypes.append('L')
    constrainsRightParts.append(1.0)

    maxCliqueModel.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=variables, val=coef)],
        rhs=constrainsRightParts,
        names=constrainsNames,
        senses=constrainsTypes)

    maxCliqueModel.set_log_stream(None)
    maxCliqueModel.set_warning_stream(None)
    maxCliqueModel.set_results_stream(None)
    return constrainsName

def checkSolution(grathNeighbors, decision, maxCliqueModel):
    clickIndex = []
    constrains = []
    constrainsNames = []
    constrainsTypes = []
    constrainsRightParts = []
    isClick = True

    for i in range(len(decision)):
        if decision[i] == 1:
            clickIndex.append(i)
    for i in range(len(clickIndex)):
        for j in range(len(clickIndex)):
            if i !=j:
                if j not in grathNeighbors[i]:
                    isClick = False
                    constrains.append([["y" + str(i), "y" + str(j)], [1, 1]])
                    constrainsNames.append("constraint_" + str(i) + "_" + str(j))
                    constrainsTypes.append('L')
                    constrainsRightParts.append(1.0)

                    maxCliqueModel.linear_constraints.add(
                        lin_expr=constrains,
                        rhs=constrainsRightParts,
                        names=constrainsNames,
                        senses=constrainsTypes)
                    break
    return isClick

# BNС
def BNC(bestDecisionValue, maxCliqueModel, return_dict):
    global bestDecision
    global grath
    global delta
    global coloredEdValue
    try:
        currentDecisionValue = solveWithCPLX(maxCliqueModel)
    except:
        print('got error1 ', bestDecision)
        return bestDecision, bestDecisionValue

    currentDecision = 0

    N = len(currentDecisionValue)
    for i in range(N):
        currentDecision = currentDecision + currentDecisionValue[i]

    if math.floor(currentDecision + delta) <= bestDecision:
        return bestDecision, bestDecisionValue

    # NEW BNC HERE
    sumMax, indSetMax = indSetSearch(copy.copy(grath), copy.copy(currentDecisionValue), copy.copy(coloredEdValue))

    while sumMax > 1 and len(indSetMax) > 0:
        constrainName = addComplexConstrain(maxCliqueModel, indSetMax)
        try:
            currentDecisionValue = solveWithCPLX(maxCliqueModel)
        except:
            removeConstrain(constrainName, maxCliqueModel, True)
            print('got error2 ', bestDecision)
            # return bestDecision, bestDecisionValue

        currentDecision = 0
        for i in range(N):
            currentDecision = currentDecision + currentDecisionValue[i]
        if math.floor(currentDecision + delta) <= bestDecision:
            return bestDecision, bestDecisionValue

        #
        # index, flag = branching(currentDecisionValue)
        #
        # if index >= N - 1 and flag == False:
        #     checkSolution(grath, currentDecisionValue, maxCliqueModel)
        #     print('current ', currentDecision)
        #     if currentDecision > bestDecision:
        #         print('new bestDecision ', bestDecision, currentDecision)
        #         bestDecision = currentDecision
        #         bestDecisionValue = currentDecisionValue
        #
        #         return_dict['power'] = bestDecision
        #         return_dict['values'] = bestDecisionValue
        #     return bestDecision, bestDecisionValue
        #
        sumMax, indSetMax = indSetSearch(grath, currentDecisionValue, copy.copy(coloredEdValue))
    # NEW BNC END
    index, flag = branching(currentDecisionValue)

    if index >= N - 1 and flag == False:
        isClick = checkSolution(grath, currentDecisionValue, maxCliqueModel)
        if isClick == True:
            print('current ', currentDecision)
            if currentDecision > bestDecision:
                print('new bestDecision ', bestDecision, currentDecision)
                bestDecision = currentDecision
                bestDecisionValue = currentDecisionValue

                return_dict['power'] = bestDecision
                return_dict['values'] = bestDecisionValue
        else:
            BNC(bestDecisionValue, maxCliqueModel, return_dict)
        return bestDecision, bestDecisionValue

    # print('index ', index, len(currentDecisionValue))
    addConstrain(index, 1, maxCliqueModel)
    BNC(bestDecisionValue, maxCliqueModel, return_dict)

    removeConstrain(index, maxCliqueModel, False)

    addConstrain(index, 0, maxCliqueModel)

    BNC(bestDecisionValue, maxCliqueModel, return_dict)

    removeConstrain(index, maxCliqueModel, False)
    return bestDecision, bestDecisionValue

def getMaxNeighbors(grath):
    maxNeighborsTmp = []
    maxNeighborsValue = -1
    for i in range(len(grath)):
        if len(grath[i]) == maxNeighborsValue:
            maxNeighborsTmp.append(i)
        if len(grath[i]) > maxNeighborsValue:
            maxNeighborsTmp = [i]
            maxNeighborsValue = len(grath[i])
    return maxNeighborsTmp

# MAIN
if __name__ == '__main__':

    bncStartEngine(localPaths)
