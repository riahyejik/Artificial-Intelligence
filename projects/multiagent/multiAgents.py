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

from game import Agent

import sys


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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        containGhost = successorGameState.getGhostPositions()
        ghost_distance = sys.float_info.max


        for j in range(len(containGhost)):
            m_distance = manhattanDistance(newPos, containGhost[j])
            if m_distance < ghost_distance:
                ghost_distance = m_distance

    	if action == 'Stop' or ghost_distance < 2:
        	return -sys.float_info.max

        containFood = newFood.asList()
        min_distance = sys.float_info.max

        for i in range(len(containFood)):
            m_distance = manhattanDistance(newPos, containFood[i])
            if m_distance <= min_distance:
                min_distance = m_distance

        utility = 0
        currentFood = currentGameState.getFood().asList()
        if newPos in currentFood:
            utility += 30
        return 1.0/min_distance + utility


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        def maxvalue(state, depth, agentindex):

            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            v = -sys.float_info.max
            legalMoves = state.getLegalActions(agentindex)
            for move in legalMoves:
                v = max(v, minvalue(state.generateSuccessor(0, move), depth, 1))
            return v

        def minvalue(state, depth, agentindex):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            v = sys.float_info.max
            legalMoves = state.getLegalActions(agentindex)
            if agentindex == num_ghosts:
                for move in legalMoves:
                    v = min(v, maxvalue(state.generateSuccessor(agentindex, move), depth + 1, 0))
            else:
                for move in legalMoves:
                    v = min(v, minvalue(state.generateSuccessor(agentindex, move), depth, agentindex + 1))
            return v

        legalMoves = gameState.getLegalActions()
        num_ghosts = gameState.getNumAgents() - 1
        direction = Directions.STOP
        max_score = -sys.float_info.max

        for move in legalMoves:
            v = minvalue(gameState.generateSuccessor(0, move), 0, 1)
            if v > max_score:
                max_score = v
                direction = move

        return direction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxvalue(state, depth, alpha, beta, agentindex):

            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            v = -sys.float_info.max
            legalMoves = state.getLegalActions(agentindex)
            for move in legalMoves:
                v = max(v, minvalue(state.generateSuccessor(0, move), depth, alpha, beta, 1))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def minvalue(state, depth, alpha, beta, agentindex):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            v = sys.float_info.max
            legalMoves = state.getLegalActions(agentindex)
            if agentindex == num_ghosts:
                for move in legalMoves:
                    v = min(v, maxvalue(state.generateSuccessor(agentindex, move), depth + 1,
                                        alpha, beta, 0))
                    if v < alpha:
                        return v
                    beta = min(beta, v)
            else:
                for move in legalMoves:
                    v = min(v, minvalue(state.generateSuccessor(agentindex, move), depth,
                                        alpha, beta, agentindex + 1))
                    if v < alpha:
                        return v
                    beta = min(beta, v)
            return v

        legalMoves = gameState.getLegalActions(0)
        num_ghosts = gameState.getNumAgents() - 1
        direction = Directions.STOP
        alpha = -sys.float_info.max
        beta = sys.float_info.max
        max_score = -sys.float_info.max

        for move in legalMoves:
            v = minvalue(gameState.generateSuccessor(0, move), 0, alpha, beta, 1)
            if v > max_score:
                max_score = v
                direction = move
            if v > beta:
                return direction
            alpha = max(alpha, v)
        return direction


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
        def maxvalue(state, depth, agentindex):

            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            v = -sys.float_info.max
            legalMoves = state.getLegalActions(agentindex)
            for move in legalMoves:
                v = max(v, expectvalue(state.generateSuccessor(0, move), depth, 1))
            return v

        def expectvalue(state, depth, agentindex):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            v = 0
            legalMoves = state.getLegalActions(agentindex)
            p = 1.0/len(legalMoves)
            if agentindex == num_ghosts:
                for move in legalMoves:
                  v += p*maxvalue(state.generateSuccessor(agentindex, move), depth + 1, 0)
            else:
                for move in legalMoves:
                  v += p*expectvalue(state.generateSuccessor(agentindex, move), depth, agentindex + 1)
            return v

        legalMoves = gameState.getLegalActions()
        num_ghosts = gameState.getNumAgents() - 1
        direction = Directions.STOP
        max_score = -sys.float_info.max

        for move in legalMoves:
            v = expectvalue(gameState.generateSuccessor(0, move), 0, 1)
            if v > max_score:
                max_score = v
                direction = move

        return direction


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>

      First I calculated the distance to closest ghost. Then for every ghost state,
      I calculated the distance to the ghost from current position. If the ghost is 
      scared, I added 400/distance to the score. If the ghost is not scared and the
      ghost is less than 2 steps away from the current position, I subtracted 200 from
      the score to prevent pacman dying. Then I calculated the minimum distance to a 
      food and used reciprocal of that and added to a score.
    """
    score = currentGameState.getScore()
    containGhost = currentGameState.getGhostPositions()
    curPos = currentGameState.getPacmanPosition()
    ghost_distance = sys.float_info.max
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    #Distance to closest ghost
    for j in range(len(containGhost)):
      m_distance = manhattanDistance(curPos, containGhost[j])
      if m_distance < ghost_distance:
        ghost_distance = m_distance

    for ghost in newGhostStates:
      distance = manhattanDistance(curPos, newGhostStates[0].getPosition())
      if ghost.scaredTimer > 0:
        score += 400.0/distance
      else:
        if ghost_distance < 2:
          score -= 200.0

    GhostStates = currentGameState.getGhostStates()
    containFood = currentGameState.getFood().asList()
    min_distance = sys.float_info.max

    #Distance to closest food
    for i in range(len(containFood)):
      m_distance = manhattanDistance(curPos, containFood[i])
      if m_distance <= min_distance:
          min_distance = m_distance

    return 1.0/min_distance + score


# Abbreviation
better = betterEvaluationFunction
