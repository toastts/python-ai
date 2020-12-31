# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newScore = successorGameState.getScore()*5
        newFoodsPos = newFood.asList()
        numFood = len(currentGameState.getFood().asList())
        newNumFood = len(newFoodsPos)
        newCapsulesPos = successorGameState.getCapsules()
        numCapsules = len(currentGameState.getCapsules())
        newNumCapsules = len(newCapsulesPos)

        if numCapsules > newNumCapsules or not newCapsulesPos:
            nearestCapsuleDistance = -100
        else:
            if newCapsulesPos:
                nearestCapsuleDistance = min([manhattanDistance(capsule, newPos) for capsule in newCapsulesPos])
            else:
                nearestCapsuleDistance = 0

        if numFood > newNumFood or not newFoodsPos:
            nearestFoodsDistance = -50
        else:
            if newFoodsPos:
                # divide by two as a tiebreaker
                nearestFoodsDistance = min([manhattanDistance(food, newPos) for food in newFoodsPos])/2
            else:
                nearestFoodsDistance = 0

        if newGhostStates:
            edibleGhostDistance = None
            nonedibleGhostDistance = None
            newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
            if sum(newScaredTimes) == 0:
                nonedibleGhostDistance = min([manhattanDistance(ghost.getPosition(), newPos) for ghost in newGhostStates])
                if nonedibleGhostDistance == 1:
                    return -999999999
            else:
                currNumberOfGhosts = len(currentGameState.getGhostStates())
                newNumberOfGhosts = len(newGhostStates)
                for ghost in newGhostStates:
                    ghostDistance = manhattanDistance(ghost.getPosition(), newPos)
                    if ghost.scaredTimer <= 1:
                        if ghostDistance == 1:
                            return -999999999
                        if nonedibleGhostDistance == None or nonedibleGhostDistance > ghostDistance:
                            nonedibleGhostDistance = ghostDistance
                    else:
                        if currNumberOfGhosts > newNumberOfGhosts:
                            edibleGhostDistance = -50
                        elif edibleGhostDistance == None or edibleGhostDistance > ghostDistance:
                            edibleGhostDistance = ghostDistance

        if edibleGhostDistance != None:
            newScore -= edibleGhostDistance
        if nonedibleGhostDistance != None:
            newScore += math.log1p(1+nonedibleGhostDistance)

        if action == 'Stop':
            if newScore > 0:
                newScore /=2
            else:
                newScore *=2

        return newScore - (nearestFoodsDistance + nearestCapsuleDistance)


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        currState = gameState
        possibleMoves = currState.getLegalActions(0)
        possibleStates = [currState.generateSuccessor(0, action) for action in possibleMoves]
        possibleScores =[self.treeifyScores(state, 1, 0) for state in possibleStates]
        bestScore = max(possibleScores)
        bestIndexes = [index for index in range(len(possibleScores)) if possibleScores[index] == bestScore]
        randIndex = random.choice(bestIndexes)

        return possibleMoves[randIndex]

    def treeifyScores(self, gameState,agentIndex, depth):

        if agentIndex == gameState.getNumAgents():
            depth +=1

        agentIndex = agentIndex % gameState.getNumAgents()

        if depth == self.depth or not gameState.getLegalActions(agentIndex) or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        nextStates = [gameState.generateSuccessor(agentIndex, action) for action in gameState.getLegalActions(agentIndex)]
        nextStateScores = [self.treeifyScores(state, agentIndex + 1, depth) for state in nextStates]

        if agentIndex > 0:
            """ MIN """
            return min(nextStateScores)
        else:
            """ MAX """
            return max(nextStateScores)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        currState = gameState
        possibleMoves = currState.getLegalActions(0)
        possibleScores = []
        alpha = -99999999999
        beta = 99999999999
        """ MAX """
        for action in possibleMoves:
            temp = self.treeifyScores(currState.generateSuccessor(0,action), 1, 0, alpha, beta)
            possibleScores.append(temp)
            alpha = max(alpha,temp)

        bestScore = max(possibleScores)
        bestIndexes = [index for index in range(len(possibleScores)) if possibleScores[index] == bestScore]
        randomIndex = random.choice(bestIndexes)
        return possibleMoves[randomIndex]


    def treeifyScores(self, gameState, agentIndex, depth, alpha, beta):

        if agentIndex == gameState.getNumAgents():
            depth += 1

        agentIndex = agentIndex % gameState.getNumAgents()

        legalActions = gameState.getLegalActions(agentIndex)

        if depth == self.depth or not legalActions or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agentIndex > 0:
            """ MIN """
            temp = 99999999999
            for action in legalActions:
                temp = min(temp, self.treeifyScores(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth, alpha, beta))
                if temp < alpha:
                    return temp
                beta = min(beta, temp)
            return temp

        else:
            """ MAX """
            temp = -99999999999
            for action in legalActions:
                temp = max(temp, self.treeifyScores(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth, alpha, beta))
                if temp > beta:
                    return temp
                alpha = max(alpha,temp)
            return temp


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        currState = gameState
        possibleMoves = currState.getLegalActions(0)
        possibleStates = [currState.generateSuccessor(0, action) for action in possibleMoves]
        possibleScores = [self.treeifyScores(state, 1, 0) for state in possibleStates]
        bestScore = max(possibleScores)
        bestIndexes = [index for index in range(len(possibleScores)) if possibleScores[index] == bestScore]
        randomIndex = random.choice(bestIndexes)

        return possibleMoves[randomIndex]

    def treeifyScores(self, gameState, agentIndex, depth):

        if agentIndex == gameState.getNumAgents():
            depth += 1

        agentIndex = agentIndex % gameState.getNumAgents()

        if depth == self.depth or not gameState.getLegalActions(
                agentIndex) or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        nextStates = [gameState.generateSuccessor(agentIndex, action) for action in
                      gameState.getLegalActions(agentIndex)]
        nextStateScores = [self.treeifyScores(state, agentIndex + 1, depth) for state in nextStates]

        if agentIndex > 0:
            return sum(nextStateScores) / float(len(nextStateScores))

        else:
            return max(nextStateScores)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currScore = currentGameState.getScore() * 5

    if currentGameState.isWin():
        return 999999999999999

    if currentGameState.isLose():
        if currScore > 0:
            return -999999999999999+currScore
        else:
            return -999999999999999-currScore

    currPos = currentGameState.getPacmanPosition()
    currFoods = currentGameState.getFood()


    currFoodsPos = currFoods.asList()
    currCapsulesPos = currentGameState.getCapsules()
    if currCapsulesPos:
        nearestCapsuleDistance = min([manhattanDistance(capsule, currPos) for capsule in currCapsulesPos])
    else:
        nearestCapsuleDistance = 0
    nearestFoodsDistance = min([manhattanDistance(food, currPos) for food in currFoodsPos]) / 2

    edibleGhostDistance = None
    nonedibleGhostDistance = None
    currGhostStates = currentGameState.getGhostStates()

    if currGhostStates:
        currScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]
        if sum(currScaredTimes) == 0:
            nonedibleGhostDistance = min([manhattanDistance(ghost.getPosition(), currPos) for ghost in currGhostStates])
        else:
            for ghost in currGhostStates:
                ghostDistance = manhattanDistance(ghost.getPosition(), currPos)
                if ghost.scaredTimer <= 1:
                    if nonedibleGhostDistance == None or nonedibleGhostDistance > ghostDistance:
                        nonedibleGhostDistance = ghostDistance
                else:
                    if edibleGhostDistance == None or edibleGhostDistance > ghostDistance:
                        edibleGhostDistance = ghostDistance

    if edibleGhostDistance != None:
        currScore -= edibleGhostDistance
    if nonedibleGhostDistance != None:
        currScore += math.log1p(1 + nonedibleGhostDistance)

    return currScore - (nearestFoodsDistance + nearestCapsuleDistance)


# Abbreviation
better = betterEvaluationFunction

