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
    # print self.index
    v = Vnminimax(gameState, self.depth, self.index)
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

    # print self.index

    v = Vnminimax(gameState, self.depth, self.index, float('-inf'), float('inf'))
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

        if player != 0:
            chosenScore = sum(scores)/float(len(scores))
            chosenMoves = legalMoves
            return (chosenScore, random.choice(chosenMoves))

        chosenScore = max(scores)
        chosenMoves = [legalMoves[index] for index in range(len(scores)) if scores[index] == chosenScore]
        curDir = state.data.agentStates[0].getDirection()
        if curDir in chosenMoves and curDir in [Directions.NORTH, Directions.SOUTH]:
            return (chosenScore, curDir)
        for move in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]:
            if move not in chosenMoves:
                continue
            return (chosenScore, move)
    # print self.index, "index"
    v = Vnminimax(gameState, self.depth, self.index)
    return v[1]
    # END_YOUR_CODE

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

dfs_time = 0
cc_time = 0
dis_time = 0
iters = 0
def betterEvaluationFunction(currentGameState):
    """
    Your extreme, unstoppable evaluation function (problem 4).

    DESCRIPTION: <write something here so we know what you did>
    """

  # BEGIN_YOUR_CODE (our solution is 15 lines of code, but don't worry if you deviate from this)
    # return currentGameState.getScore() # +100 - 20*len(currentGameState.getCapsules())
    from collections import deque
    global dfs_time, cc_time, iters, dis_time
    myPos = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    walls = currentGameState.getWalls()

    w = foods.width
    h = foods.height

    dpos = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    pos_to_comp = {}
    comp_to_pos = {}
    oddy_nodes = {}
    class Config(object):
        def __init__(self):
            self.num_comps=0
    config = Config()

    def SetIfOddy(x, y, comp_id):
        if (walls[x-1][y] != walls[x+1][y]) or (walls[x][y-1] != walls[x][y+1]):
            oddy_nodes[comp_id].add((x, y))
            return True
        return False
        numfoods = 0
        numwalls = 0
        for dx, dy in dpos:
            xi = x+dx
            yi = y+dy
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                continue
            if foods[xi][yi]:
                numfoods += 1
            if walls[xi][yi]:
                numwalls +=1
        if numfoods != 2 or numwalls != 2:
            oddy_nodes[comp_id].add((xi, yi))

    def assignComponent(x, y, comp_id, maxcnt=100):
        q= deque()
        pos_to_comp[(x, y)] = comp_id
        comp_to_pos[comp_id].add((x, y))
        SetIfOddy(x, y, comp_id)

        q.append((x, y))
        while len(q) > 0 and maxcnt > 0:
            xi, yi = q.popleft()
            maxcnt -= 1
            # print xi, yi
            for dx, dy in dpos:
                xi2 = xi + dx
                yi2 = yi + dy
                if not foods[xi2][yi2]:
                    continue
                pos = (xi2, yi2)
                if pos in pos_to_comp:
                    continue
                pos_to_comp[pos] = comp_id
                comp_to_pos[comp_id].add(pos)
                if SetIfOddy(xi2, yi2, comp_id):
                    q.append((xi2, yi2))


    def createComponents(start):
        q = deque()
        q.append(start)
        visited = set()
        while len(q) > 0:
            x, y = q.popleft()
            for dx, dy in dpos:
                xi = x + dx
                yi = y + dy
                if (xi, yi) in visited:
                    continue
                visited.add((xi, yi))
                if foods[xi][yi]:
                    if (xi, yi) not in pos_to_comp:
                        comp_to_pos[config.num_comps] = set()
                        oddy_nodes[config.num_comps] = set()
                        assignComponent(xi, yi, config.num_comps)
                        config.num_comps += 1
                    continue
                if walls[xi][yi]:
                    continue
                q.append((xi, yi))
        return

        for xi in range(w):
            for yi in range(h):
                if not foods[xi][yi]:
                    continue
                pos = (xi, yi)
                if pos in pos_to_comp:
                    continue
                comp_to_pos[config.num_comps] = set()
                oddy_nodes[config.num_comps] = set()
                assignComponent(xi, yi, config.num_comps)
                config.num_comps += 1
        # print(config.num_comps)

    def getCompEnds(comp_id):
        ends = oddy_nodes[comp_id]
        if len(ends) == 0:
            return comp_to_pos[comp_id]
        return ends

    import time
    cc_stime = time.time()
    # createComponents(myPos)
    cc_time += time.time() - cc_stime

    def findDis(starts, ends, _ = False):
        global dis_time
        dis_stime = time.time()
        mindis = 1000000
        minp = None
        scnt=0
        for s in starts:
            for e in ends:
                dis = abs(s[0] - e[0]) + abs(s[1] - e[1])
                if dis < mindis:
                    mindis = dis
                    minp = e
        dis_time += time.time() - dis_stime
        return (mindis, minp)

    def findDisDFS(starts, ends):
        q = deque()
        for s in starts:
            q.append((s, 0))
        visited = set()
        while len(q) > 0:
            (x, y), dis = q.popleft()
            dis += 1
            for dx, dy in dpos:
                xi = x+dx
                yi = y+dy
                pos = (xi, yi)
                if  pos in visited:
                    continue
                if pos in ends:
                    return dis, pos
                if walls[xi][yi]:
                    continue
                q.append(((xi, yi), dis))
                visited.add(pos)
        return 100000, None


    def compDFS(starts, taken_comps):
        if len(taken_comps) == config.num_comps:
            return 0

        min_cost = float('inf')
        num_tried = 0
        disFn = findDisDFS if len(taken_comps) == 0 else findDis
        for comp_id in range(config.num_comps):
            if comp_id in taken_comps:
                continue
            num_tried += 1
            if num_tried > 2:
                break
            ends = getCompEnds(comp_id)
            disCost, endPoint = disFn(starts, ends)

            # TODO: maybe remove chosen one from ends 

            # print "disCost", disCost
            taken_comps.add(comp_id)
            if len(ends) > 0 and endPoint is not None:
                ends.remove(endPoint)
            cost = disCost # + compDFS(ends, taken_comps) # + len(oddy_nodes[comp_id])
            taken_comps.remove(comp_id)
            if endPoint is not None:
                ends.add(endPoint)

            if cost < min_cost:
                min_cost = cost
        return min_cost

    def findNearestFood(starts):
        q = deque()
        for s in starts:
            q.append((s, 0))
        visited = set()
        while len(q) > 0:
            (x, y), dis = q.popleft()
            dis += 1
            for dx, dy in dpos:
                xi = x+dx
                yi = y+dy
                pos = (xi, yi)
                if pos in visited or walls[xi][yi]:
                    continue
                if foods[xi][yi]:
                    return dis, pos
                visited.add(pos)
                q.append(((xi, yi), dis))
        return 100000, None


    dfs_stime = time.time()
    # totalCost = compDFS([myPos], set()) + 40*(len(currentGameState.getCapsules()))
    totalCost = findNearestFood([myPos])[0] + 40*(len(currentGameState.getCapsules()))
    dfs_time += time.time() - dfs_stime
    iters += 1
    # if iters % 10000 == 0:
        # print dfs_time, cc_time, dis_time

    for agent in currentGameState.data.agentStates:
        if agent.scaredTimer > 0:
            agentDis, _ = findDis([myPos], [agent.getPosition()])
            # print "agent", agent.getPosition(), agentDis, agent.scaredTimer
            if  agentDis < agent.scaredTimer:
                totalCost = -50 + agentDis
                break
    # print totalCost
    # print myPos, totalCost, config.num_comps, currentGameState.getScore()
    # print comp_to_pos
    # print oddy_nodes
    return 8*len(foods.asList()) - totalCost + currentGameState.getScore()


  # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction
