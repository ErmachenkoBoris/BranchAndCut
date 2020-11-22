import multiprocessing
import urllib.request
import time
import cplex
import random
import math
import numpy as np

# GLOBAL
timeLimitValue = 3600 * 3
delta = 0.00001
local = True
bestDecision = 0

paths = [
      'http://iridia.ulb.ac.be/~fmascia/files/DIMACS/C125.9.clq',
    'http://iridia.ulb.ac.be/~fmascia/files/DIMACS/brock200_2.clq',
     'http://iridia.ulb.ac.be/~fmascia/files/DIMACS/keller4.clq'
]

localPaths = [
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/c-fat200-1.clq',
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/c-fat200-2.clq',
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/c-fat200-5.clq',
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/c-fat500-1.clq',
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/c-fat500-10.clq',
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/c-fat500-2.clq',
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/c-fat500-5.clq',
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/MANN_a9.clq',
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/hamming6-2.clq',
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
#            [1,1,0,0],
#           [1,1,1,1],
#           [0,1,1,1],
#           [0,1,1,1]])
matrixTest = np.array([
            [1,1,1,1],
           [1,1,1,1],
          [1,1,1,1],
    [1,1,1,1]])
# -----------------------------------------------TEST-------------------------------


#--------------------OPEN FILE--------------------
def openGraph(filePath):
    n = -1
    m = -1
    global local
    if local==True:
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


#---------------------------------------------------------------------
#----------------------Heuristic Functions--------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
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
        while len(matrix[i].nonzero()[0]) != edges: # пока есть ненулевые
            for j in range(i, edges): # ?
                if matrix[i, j] == 0 and j in V:
                    break
            if j == edges:
                break
            if j == edges-1 and matrix[i, j] != 0 or j not in V:
                break

            colorGroups[k].append(j)
            V.remove(j)
            matrix[i] = matrix[i] + matrix[j]

        k = k+1
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
            if matrix[i,j] == 1 and i!=j and coloredEdges[j] not in colorTmpCounter:
                colorTmpCounter.append(coloredEdges[j])
        if len(colorTmpCounter) == maxColorCount:
            maxColorCountEdges.append(i)
        if len(colorTmpCounter) > maxColorCount:
            maxColorCount = len(colorTmpCounter)
            maxColorCountEdges = [i]
    return maxColorCountEdges

# эвристический поиск клики (с помощью раскрашенного графа)
def findEvristicClique(maxColorCandidatARRAY, matrix, n, coloredEd):
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

    if(len(clickCandidat) > 0):
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

       clickEvrF.append(maxColorLocal) # добавили в клику

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
def evristic(path):

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

    return bestEvrValue, clickValue, confusion_matrix, n
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------




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

    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            if matrix[i][j] == 0.0:
                constrains.append([["y" + str(i), "y" + str(j)], [1, 1]])
                constrainsNames.append("constraint_" + str(i) + "_" + str(j))
                constrainsTypes.append('L')
                constrainsRightParts.append(1.0)

    maxCliqueModel.linear_constraints.add(
        lin_expr=constrains,
        rhs=constrainsRightParts,
        names=constrainsNames,
        senses=constrainsTypes)

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

# ------------------------BNB----------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
# функция, которую запускаем в отедльном процессе, чтобы была возможность остановить по времени
def bnbContainer(evristicPower, evristicValues, matrix, n, graph, return_dict):
    start_BNB_time = time.time()
    constrains = []
    constrainsNames = []
    constrainsTypes = []
    constrainsRightParts = []

    global bestDecision

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

    result, resultValues = BNB(evristicValues, maxCliqueModel, return_dict)
    print('grapth NAME ', graph)
    print("--- %s seconds BNB ---" % (time.time() - start_BNB_time))
    print('!!!!! result ', result)
    print('!!!!! resultValues ', resultValues)

#     проход по всем графам из файлов и запуск эвристики и bnb для каждого
def bnbStartEngine(graphs):
    for i in range(len(graphs)):
        evristicPower, evristicValues, matrix, n = evristic(graphs[i])
        print('new ittt')
        if __name__ == '__main__':
            global timeLimitValue
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            return_dict['power'] = 0
            return_dict['values'] = []
            print('start bnb for ', graphs[i])
            p = multiprocessing.Process(target=bnbContainer, args=(evristicPower, evristicValues, matrix, n, graphs[i], return_dict))
            p.start()
            p.join(timeLimitValue)
            if p.is_alive():
                p.terminate()
                print("--- %s seconds LIMIT BNB ---" % timeLimitValue)
                print('!!!!! TIMEOUT result ', return_dict['power'] )
                print('!!!!! TIMEOUT resultValues ', return_dict['values'])

# округление с учетом дельты
def numberWithDelta(number, maxValue, minVale, eps):
    if number + eps >= maxValue:
        return maxValue
    if number - eps <= minVale:
        return minVale
    return number

# добавление ограничения
def addConstrain(i,value,maxCliqueModel):
    maxCliqueModel.linear_constraints.add(
        lin_expr=[[["y" + str(i)], [1]]],
        rhs=[value],
        names=["constraint_" + str(i)],
        senses=['E']
    )

# удаление ограничения
def removeConstrain(i,maxCliqueModel):
    maxCliqueModel.linear_constraints.delete("constraint_" + str(i))

# решение сиплексом
def solveWithCPLX(maxCliqueModel):
    maxCliqueModel.solve()
    return maxCliqueModel.solution.get_values()

# BNB
def BNB(bestDecisionValue, maxCliqueModel, return_dict):
    global bestDecision
    currentDecisionValue = solveWithCPLX(maxCliqueModel)
    N = len(currentDecisionValue)

    currentDecision = 0

    for i in range(N):
        currentDecision = currentDecision + currentDecisionValue[i]

    global delta
    if math.floor(currentDecision + delta) <= bestDecision:
        return bestDecision, bestDecisionValue

    flag = False
    for index in range(N):
        currentDecisionValue[index] = numberWithDelta(currentDecisionValue[index], 1, 0, delta)
        if currentDecisionValue[index] !=0 and currentDecisionValue[index] !=1:
            flag = True
            break
    if index == N - 1 and flag == False:
        if currentDecision > bestDecision:
            print('new bestDecision ', bestDecision, currentDecision)

            bestDecision = currentDecision
            bestDecisionValue = currentDecisionValue

            return_dict['power'] = bestDecision
            return_dict['values'] = bestDecisionValue

        return bestDecision, bestDecisionValue


    addConstrain(index, 1, maxCliqueModel)

    BNB(bestDecisionValue, maxCliqueModel, return_dict)

    removeConstrain(index, maxCliqueModel)

    addConstrain(index, 0, maxCliqueModel)

    BNB(bestDecisionValue, maxCliqueModel, return_dict)

    removeConstrain(index, maxCliqueModel)

    return bestDecision, bestDecisionValue

# MAIN
if __name__ == '__main__':
    bnbStartEngine(localPaths)
    # bnbStartEngine(paths)