import multiprocessing
import urllib.request
import time
import cplex
import random
import math
import numpy as np
import copy

# GLOBAL
timeLimitValue = 3600
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
    'C:/Users/cbkf1/PycharmProjects/BranchAndBound/graphs/san200_0.7_2.clq',
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
matrixTest = np.array([
    [1, 1, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 1, 1]])



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
            if i != j and graphMatrix[i][j] == 1 and j not in graphModel[i]:
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

# находим для каждого узла сколько у него разноцветных соседей
def getColoredNumber(matrix, n, coloredEdges):
    vertexWithColoredPower = [0 for i in range(n)]
    for i in range(n):
        colorTmpCounter = []
        for j in range(n):
            if matrix[i, j] == 1 and i != j and coloredEdges[j] not in colorTmpCounter:
                colorTmpCounter.append(coloredEdges[j])
        vertexWithColoredPower[i] = len(colorTmpCounter)
    return vertexWithColoredPower

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
        if len(colorTmpCounter) == maxColorCount and len(maxColorCountEdges) > 0:
            maxColorCountEdges.append(i)
        if len(colorTmpCounter) > maxColorCount:
            maxColorCount = len(colorTmpCounter)
            maxColorCountEdges = [i]
    return maxColorCountEdges


# эвристический поиск клики (с помощью раскрашенного графа)
def findEvristicClique(maxColorCandidatARRAY, matrix, n, coloredPower):
    clickCandidat = []
    # Проверим, вдруг мы взяли клику из трех элементов, тогда начнем с нее
    if matrix[maxColorCandidatARRAY[0], maxColorCandidatARRAY[1]] == matrix[maxColorCandidatARRAY[1], maxColorCandidatARRAY[2]] == matrix[maxColorCandidatARRAY[0], maxColorCandidatARRAY[2]] == 1:
        lengthData = len(maxColorCandidatARRAY)
        listOfNeiborsList = [[] for l in range(lengthData)]
        for j in range(lengthData):
            for i in range(n):
                if matrix[maxColorCandidatARRAY[j]][i] == 1 and i != maxColorCandidatARRAY[j]:
                    listOfNeiborsList[j].append(i)
        clickCandidat = list(set(listOfNeiborsList[0]) & set(listOfNeiborsList[1]) & set(listOfNeiborsList[2]))
    if(len(clickCandidat) > 0):
        clickEvr = copy.copy(maxColorCandidatARRAY)
    else:
        clickEvr = [random.choice(maxColorCandidatARRAY)]
        for i in range(n):
            if matrix[clickEvr[0]][i] == 1 and i != clickEvr[0]:
                clickCandidat.append(i)

    def findClickEvr(clickEvrF, clickCandidatF, matrix):
       maxColorLocal = random.choice(clickCandidatF)
       clickEvrF.append(maxColorLocal) # добавили в клику

       clickLocalCandidat = []  # ищем соседей новых
       for i in range(n):
           if matrix[maxColorLocal, i] == 1 and i != maxColorLocal and i not in clickEvrF:
               clickLocalCandidat.append(i)

        # находим пересечение со старыми соседями
       newCandidats = list(set(clickCandidatF) & set(clickLocalCandidat))
       if len(newCandidats) > 0:
           return findClickEvr(clickEvrF, newCandidats, matrix)
       return clickEvrF

    return findClickEvr(clickEvr, clickCandidat, matrix)

# запуск всего механизма эвристики
def evristic(matrix, n, path):
    print('heuristics start for: ', path)
    n, m, confusion_matrix = openGraph(path)

    start_evr_time = time.time()

    coloredEd = colorGreedy(confusion_matrix.copy(), n)
    coloredPower = getColoredNumber(matrix, n, coloredEd)
    maxColor = getWithMaxColorNumber(confusion_matrix.copy(), n, coloredEd)

    bestEvrValue = -1
    bestEvrStore = []

    for i in range(5000):
        randomEdges = random.sample(maxColor, 3)
        clickEvristic = findEvristicClique(randomEdges, confusion_matrix.copy(), n, coloredPower)
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

    print('--- grath Name: ', path)  # Название графа
    print("--- seconds Heuristics: %s" % (time.time() - start_evr_time))  # время эвристики
    print('--- colors of vertexes: ', coloredEd)  # раскраска графа
    print('--- maxColor connected vertex: ', maxColor)  # узлы с наибольшим количеством разноцветных соседей
    print('--- Heuristics values: ', bestEvrFinal)  # решение клики
    print('--- Heuristics Power: ', bestEvrValue)  # количество узлов в клике
    print('')
    coloredEdValue = coloredEd

    return bestEvrValue, clickValue, matrix, n, coloredEdValue, bestEvrFinal

# ---------------------------------------------------------------------
def indSetSearch(neighborsGraph, weight, coloredEd):
    indSet = evristicMaxIndSetSearch(coloredEd, weight)
    sumMax = 0
    indSetMax = []

    for i in range(50):
        sum, indSetNew = localSearch(neighborsGraph, copy.copy(indSet), copy.copy(weight))
        if sumMax < sum:
            sumMax = sum
            indSetMax = copy.copy(indSetNew)

    for i in indSetMax:
        for j in indSetMax:
            if i != j and i in neighborsGraph[j]:
                print('ERROR IND SET 2', indSet, indSetMax)
                break
    return sumMax, indSetMax

def evristicMaxIndSetSearch(colored, weigth):
    indeSetSumMax = -1
    indSetMax = []
    for z in range(len(colored)):

        randomColor = colored[z]
        indSetSum = 0;
        indSet = []
        for i in range(len(colored)):
            if colored[i] == randomColor:
                indSetSum=indSetSum + weigth[i]
                indSet.append(i)
        if indeSetSumMax < indSetSum:
            indeSetSumMax = indSetSum
            indSetMax = copy.copy(indSet)
    return indSetMax

# локальный поиск
def localSearch(graphNeighbors, indSet, weight):
    indSetOld = copy.copy(indSet)

    N = len(graphNeighbors)
    # statusArray:
    # 1 - indSet
    # 2 - freeVertex
    # 3 - bindedVertex
    tightness = [0 for i in range(N)]
    statusArray = [2 for i in range(N)]
    for i in indSet:
            statusArray[i] = 1
            for j in graphNeighbors[i]:
                tightness[j] += 1
                statusArray[j] = 3
    candidatsVertex = []
    for item in indSet:
        tightCount = 0
        for j in graphNeighbors[item]:
            if tightness[j] == 1:
                tightCount += 1
                if tightCount >= 2:
                    candidatsVertex.append(item)
                    break
    freeVertex = []
    for i in range(len(statusArray)):
        if statusArray[i] == 2:
            freeVertex.append(i)

    while len(candidatsVertex) > 0 or len(freeVertex)>0:
        if len(candidatsVertex) == 0:
            newVertex = random.choice(freeVertex)
            candidatsVertex.append(newVertex)
            statusArray[newVertex] = 1
            freeVertex.remove(newVertex)
            for val in graphNeighbors[newVertex]:
                tightness[val] += 1
        vertForSwap = random.choice(candidatsVertex)
        u = -1
        v = -1
        for i in graphNeighbors[vertForSwap]:
            if tightness[i] == 1:
                if u == -1:
                    u = i
                else:
                    if i not in graphNeighbors[u]:
                        v = i
            if u != -1 and v != -1:
                break
        if u != -1 and v != -1:
            statusArray[u] = statusArray[v] = 1
            statusArray[vertForSwap] = 3

            candidatsVertex.remove(vertForSwap)
            for j in graphNeighbors[vertForSwap]:
                tightness[j] = tightness[j]- 1

            for j in graphNeighbors[u]:
                tightness[j] += 1


            for j in graphNeighbors[v]:
                tightness[j] += 1

            for j in range(len(statusArray)):
                if tightness[j] <= 0 and statusArray[j] != 1:
                    statusArray[j] = 2
                if tightness[j] >= 1:
                    statusArray[j] = 3
            candidatsVertex = []
            for j in range(len(statusArray)):
                if statusArray[j]==1:
                    tightCount = 0
                    for j1 in graphNeighbors[j]:
                        if tightness[j1] == 1:
                            tightCount += 1
                            if tightCount >= 2:
                                candidatsVertex.append(j)
                                break
            freeVertex = []
            for j in range(len(statusArray)):
                if statusArray[j] == 2 and tightness[j] == 0:
                    freeVertex.append(j)
        else:
            freeVertex = []
            for j in range(len(statusArray)):
                if statusArray[j] == 2 and tightness[j] == 0:
                    freeVertex.append(j)
            candidatsVertex.remove(vertForSwap)
            continue
    sumWeight = 0
    answer = []
    sumWeightOld = 0
    for i in indSetOld:
        sumWeightOld = sumWeightOld + weight[i]
    for i in range(len(statusArray)):
        if statusArray[i] == 1:
            sumWeight += weight[i]
            answer.append(i)
    if (sumWeightOld> sumWeight):
        return sumWeightOld, indSetOld
    return sumWeight, answer

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

# Инициализируем модель cplex (добавляем все ограничения)
def initalClickCPLEX(maxCliqueModel, n):
    maxCliqueModel.variables.add(names=["y" + str(i) for i in range(n)],
                                 types=[maxCliqueModel.variables.type.continuous for i in range(n)])

    for i in range(n):
        maxCliqueModel.variables.set_lower_bounds(i, 0.0)
        maxCliqueModel.variables.set_upper_bounds(i, 1.0)

    maxCliqueModel.set_log_stream(None)
    maxCliqueModel.set_warning_stream(None)
    maxCliqueModel.set_error_stream(None)
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
def bncContainer(evristicPower, evristicValues, n, return_dict, coloredEd, grathNeighborsV):
    start_BNC_time = time.time()

    global bestDecision
    global coloredEdValue
    global grath

    coloredEdValue = coloredEd
    grath = grathNeighborsV

    maxCliqueModel = cplex.Cplex()

    initalClickCPLEX(maxCliqueModel, n)

    bestDecision = evristicPower

    return_dict['power'] = evristicPower
    return_dict['values'] = evristicValues
    constrainStack = []
    global timer
    timer = 0
    result, resultValues = BNC(evristicValues, maxCliqueModel, return_dict, constrainStack)
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
        evristicPower, evristicValues, matrix, n, coloredEdValue, bestEvrFinal = evristic(confusion_matrix, n, graphs[i])

        # check decision
        isClick = True
        for p in bestEvrFinal:
            for j in bestEvrFinal:
                if p != j:
                    if matrix[j][p] != 1:
                        print('error ', j)
                        isClick = False
                        break
        print("CHECK EVRISTIC - ", isClick)
        #
        if __name__ == '__main__':
            global timeLimitValue
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            return_dict['power'] = 0
            return_dict['values'] = []
            print('start bnC for ', graphs[i])
            p = multiprocessing.Process(target=bncContainer,
                                        args=(evristicPower, evristicValues, n, return_dict, coloredEdValue, grath))
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
    return "constraint_" + str(i)


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

def checkSolution(grathNeighbors, decision, maxCliqueModel, constrainStack):
    clickIndex = []
    constrains = []
    constrainsNames = []
    constrainsTypes = []
    constrainsRightParts = []
    isClick = True

    for i in range(len(decision)):
        if decision[i] == 1:
            clickIndex.append(i)
    for i in clickIndex:
        for j in clickIndex:
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

def checkForSlack(constrainStack):
    constrainsToDelete = []
    for con1 in constrainStack:
        notSlack = False
        for con2 in constrainStack:
            if con1 != con2:
                arr1 = con1.split('_')
                arr2 = con2.split('_')

                mix = list(set(arr1) & set(arr2))
                if len(mix) == len(arr1):
                    notSlack = True
                    break
        if notSlack == False:
            constrainStack.remove(con1)
            constrainsToDelete.append(con1)

    return constrainsToDelete

# BNС
def BNC(bestDecisionValue, maxCliqueModel, return_dict, constrainStack):
    global timer
    global bestDecision
    global grath
    global delta
    global coloredEdValue
    timer +=1
    try:
        currentDecisionValue = solveWithCPLX(maxCliqueModel)
    except:
        return bestDecision, bestDecisionValue

    currentDecision = 0

    N = len(currentDecisionValue)
    for i in range(N):
        currentDecision = currentDecision + currentDecisionValue[i]

    if math.floor(currentDecision + delta) <= bestDecision:
        return bestDecision, bestDecisionValue

    # NEW BNC HERE
    sumMax, indSetMax = indSetSearch(copy.copy(grath), currentDecisionValue, copy.copy(coloredEdValue))
    steps = 0
    deltaLocal = 0.5
    oldDecisionSum = copy.copy(currentDecision)
    badIterCount = 0
    while sumMax - delta> 1 and len(indSetMax) > 0:
        constrainName = addComplexConstrain(maxCliqueModel, indSetMax)

        constrainStack.append(constrainName)
        try:
            currentDecisionValueTmp = solveWithCPLX(maxCliqueModel)
            currentDecisionValue = copy.copy(currentDecisionValueTmp)
        except:
            constrainStack.remove(constrainName)
            removeConstrain(constrainName, maxCliqueModel, True)

        currentDecision = 0
        for i in range(N):
            currentDecision = currentDecision + currentDecisionValue[i]
        if math.floor(currentDecision + delta) <= bestDecision:
                return bestDecision, bestDecisionValue
        sumMax, indSetMax = indSetSearch(grath, currentDecisionValue, copy.copy(coloredEdValue))

        if abs(oldDecisionSum - currentDecision) < deltaLocal:
            badIterCount += 1
            if badIterCount > 50:
                break
        else:
            badIterCount = 0
            oldDecisionSum = copy.copy(currentDecision)

    if timer > 500:
        timer = 0
        toDelete = checkForSlack(constrainStack)
        if len(toDelete) > 0:
            for slackj in toDelete:
                maxCliqueModel.linear_constraints.delete(slackj)
    index, flag = branching(currentDecisionValue)

    if index >= N - 1 and flag == False:
        isClick = checkSolution(grath, currentDecisionValue, maxCliqueModel, constrainStack)
        if isClick == True:
            if currentDecision > bestDecision:
                print('! new bestDecision ', bestDecision, currentDecision)
                bestDecision = currentDecision
                bestDecisionValue = currentDecisionValue

                return_dict['power'] = bestDecision
                return_dict['values'] = bestDecisionValue
        else:
            BNC(bestDecisionValue, maxCliqueModel, return_dict, constrainStack)
        return bestDecision, bestDecisionValue

    constrName = addConstrain(index, 1, maxCliqueModel)
    BNC(bestDecisionValue, maxCliqueModel, return_dict, constrainStack)

    removeConstrain(constrName, maxCliqueModel, True)

    constrName = addConstrain(index, 0, maxCliqueModel)

    BNC(bestDecisionValue, maxCliqueModel, return_dict, constrainStack)

    removeConstrain(constrName, maxCliqueModel, True)
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
