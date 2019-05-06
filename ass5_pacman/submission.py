from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions():
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


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
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """

    # BEGIN_YOUR_CODE (our solution is 26 lines of code, but don't worry if you deviate from this)
    def Vnminimax(state, d, player):
        if state.isWin() or state.isLose():
            return (state.getScore(), Directions.STOP)
        if d == 0:
            return (self.evaluationFunction(state), Directions.STOP)  # Direction here will never be used.

        # Collect legal moves and successor states
        # print(player, state.getNumAgents())
        legalMoves = state.getLegalActions(player)

        nextPlayer = player + 1
        nextDepth = d
        if nextPlayer == state.getNumAgents():
            nextPlayer = 0
            nextDepth = d - 1

        # Choose one of the best actions
        scores = [Vnminimax(state.generateSuccessor(player, action), nextDepth, nextPlayer)[0] for action in legalMoves]

        choice_fn = max if player == 0 else min
        chosenScore = choice_fn(scores)
        chosenIndices = [index for index in range(len(scores)) if scores[index] == chosenScore]
        chosenIndex = random.choice(chosenIndices) # Pick randomly among the best
        return (chosenScore, legalMoves[chosenIndex])
    v = Vnminimax(gameState, self.depth, 0)
    # print(v)
    return v[1]

    # END_YOUR_CODE

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE (our solution is 49 lines of code, but don't worry if you deviate from this)
    def Vnminimax(state, d, player, alpha, beta):
        if state.isWin() or state.isLose():
            return (state.getScore(), Directions.STOP)
        if d == 0:
            return (self.evaluationFunction(state), Directions.STOP)  # Direction here will never be used.

        # Collect legal moves and successor states
        # print(player, state.getNumAgents())
        legalMoves = state.getLegalActions(player)

        nextPlayer = player + 1
        nextDepth = d
        if nextPlayer == state.getNumAgents():
            nextPlayer = 0
            nextDepth = d - 1

        # Choose one of the best actions
        if player == 0:
            bestV = float('-inf')
            bestAction = ''
            for action in legalMoves:
                v = Vnminimax(state.generateSuccessor(player, action), nextDepth, nextPlayer, alpha, beta)[0]
                if v > bestV:
                    bestV = v
                    bestAction = action
                alpha = max(alpha, bestV)
                if alpha >= beta:
                    break
            return (bestV, bestAction)
        else:
            bestV = float('inf')  # best is mininum
            bestAction = ''
            for action in legalMoves:
                v = Vnminimax(state.generateSuccessor(player, action), nextDepth, nextPlayer, alpha, beta)[0]
                if v < bestV:
                    bestV = v
                    bestAction = action
                beta = min(beta, bestV)
                if alpha >= beta:
                    break
            return (bestV, bestAction)

    v = Vnminimax(gameState, self.depth, 0, float('-inf'), float('inf'))
    print(v)
    return v[1]
    # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_CODE (our solution is 25 lines of code, but don't worry if you deviate from this)
    def Vnminimax(state, d, player):
        if state.isWin() or state.isLose():
            return (state.getScore(), Directions.STOP)
        if d == 0:
            return (self.evaluationFunction(state), Directions.STOP)  # Direction here will never be used.

        # Collect legal moves and successor states
        # print(player, state.getNumAgents())
        legalMoves = state.getLegalActions(player)

        nextPlayer = player + 1
        nextDepth = d
        if nextPlayer == state.getNumAgents():
            nextPlayer = 0
            nextDepth = d - 1

        # Choose one of the best actions
        scores = [Vnminimax(state.generateSuccessor(player, action), nextDepth, nextPlayer)[0] for action in legalMoves]

        if player == 0:
            chosenScore = max(scores)
            chosenMoves = [legalMoves[index] for index in range(len(scores)) if scores[index] == chosenScore]
        else:
            chosenScore = sum(scores)/float(len(scores))
            chosenMoves = legalMoves
        return (chosenScore, random.choice(chosenMoves))
    v = Vnminimax(gameState, self.depth, 0)
    return v[1]
    # END_YOUR_CODE

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
    Your extreme, unstoppable evaluation function (problem 4).

    DESCRIPTION: <write something here so we know what you did>
  """

  # BEGIN_YOUR_CODE (our solution is 15 lines of code, but don't worry if you deviate from this)
  from heapq import heappush, heappop
  myPos = currentGameState.getPacmanPosition()
  foods = currentGameState.getFood()
  walls = currentGameState.getWalls()
  cost = 0
  pq = [] 
  heappush(pq, (0, myPos))
  diff = [(-1, 0), (1, 0), (0, -1), (0, 1)]
  leastdis = {myPos: 0}
  pq_visited=set()

  while len(pq) > 0:
    pq_cost, pq_start = heappop(pq)
    if pq_start in pq_visited:
      continue
    pq_visited.add(pq_start)
    cost += pq_cost
    q = [(pq_start, 0)]
    q_visited = set()
    while len(q) > 0:
      q_start, dis = q[0]
      q = q[1:]
      for d in diff:
        pos = (q_start[0]+d[0], q_start[1]+d[1])
        if pos[0] <0 or pos[1] < 0 or pos[0] >= foods.width or pos[1] >= foods.height:
          continue
        if walls[pos[0]][pos[1]]:
          continue
        if pos in q_visited:
          continue
        q_visited.add(pos)
        pos_dis = dis + 1
        if foods[pos[0]][pos[1]]:
          if dis < leastdis.get(pos, 10000000):
              leastdis[pos] = dis
              heappush(pq, (pos_dis, pos))
          continue
        q.append((pos, pos_dis))
  return 10*len(foods.asList()) - 2*cost  + currentGameState.getScore()
  # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction
