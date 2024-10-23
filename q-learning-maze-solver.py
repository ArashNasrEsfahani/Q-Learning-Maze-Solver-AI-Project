import numpy as np
import random

class Vertex:
    def __init__(self, reward):
        self.reward = reward
        self.edges = []

class Edge:
    def __init__(self, destination):
        self.destination = destination

class Graph:
    def __init__(self, rows, columns):
        self.width = rows
        self.height = columns
        self.vertices = [[None for j in range(rows)] for i in range(columns)]

    def add_vertex(self, i, j, reward):
        vertex = Vertex(reward)
        self.vertices[i][j] = vertex

    def add_edge(self, source, destination):
        edge = Edge(destination)
        source.edges.append(edge)

    def initilaze_edges(self):
        for i in range (10):
            for j in range (10):
                if (i <  9):
                    self.add_edge(self.vertices[i][j], self.vertices[i+1][j])
                if(i > 0):
                    self.add_edge(self.vertices[i][j], self.vertices[i-1][j])
                if(j > 0):
                    self.add_edge(self.vertices[i][j], self.vertices[i][j-1])
                if(j < 9):
                    self.add_edge(self.vertices[i][j], self.vertices[i][j+1])

g = Graph(10, 10)


maze = np.array([["W", "B", "W", "W", "W", "W", "W", "W", "W", "W"],
                ["W", "W", "W", "F", "W", "B", "W", "W", "W", "W"],
                ["F", "W", "W", "W", "W", "B", "F", "W", "W", "W"],
                ["B", "B", "W", "B", "B", "W", "B", "W", "W", "W"],
                ["W", "F", "B", "F", "B", "F", "B", "B", "B", "W"],
                ["W", "W", "B", "W", "B", "W", "W", "W", "W", "W"],
                ["W", "W", "W", "W", "W", "W", "W", "W", "W", "W"],
                ["W", "F", "W", "W", "W", "W", "B", "B", "B", "B"],
                ["W", "B", "B", "B", "B", "B", "W", "W", "W", "W"],
                ["W", "W", "W", "F", "W", "B", "W", "B", "F", "T"]])

rewards = {'W': -0.2, 'B': -0.9, 'F': 0.5, 'T': 1}

actions = ['down', 'right', 'up', 'left']

QTable = np.zeros((maze.shape[0], maze.shape[1], len(actions)))

for i in range(10):
    for j in range(10):
            g.add_vertex(i, j, maze[i][j])

g.initilaze_edges()



def move(state, action):
    i, j = state
    if action == 'up':
        i = i - 1
    elif action == 'down':
        i = i + 1
    elif action == 'left':
        j = j - 1
    elif action == 'right':
        j = j + 1

    return (i, j)


def select_action(state, Q, epsilon = 0.2):
       
    selected = {'down': Q[state[0], state[1], 0], 'right': Q[state[0], state[1], 1], 'up': Q[state[0], state[1], 2], 'left': Q[state[0], state[1], 3]}
    if(state[0] == 0):
        selected.pop('up')
    if(state[0] == 9):
        selected.pop('down')
    if(state[1] == 0):
        selected.pop('left')
    if(state[1] == 9):
        selected.pop('right')

    action = None
    max_val = None

    if (np.random.uniform(0, 1) >= epsilon):
        for key, val in selected.items():
            if max_val is None or val > max_val:
                max_val = val
                action = key
        return action
    else:
        return random.choice(list(selected.keys()))

def select_action_final(Q, state):
    selected = {'down': Q[state[0], state[1], 0], 'right': Q[state[0], state[1], 1], 'up': Q[state[0], state[1], 2], 'left': Q[state[0], state[1], 3]}
    if(state[0] == 0):
        selected.pop('up')
    if(state[0] == 9):
        selected.pop('down')
    if(state[1] == 0):
        selected.pop('left')
    if(state[1] == 9):
        selected.pop('right')

    action = None
    max_val = None

    for key, val in selected.items():
        if max_val is None or val > max_val:
            max_val = val
            action = key
    return action

def reward_calc(maze, nextState, rewards, visitedFlags):
    if(maze[nextState[0]][nextState[1]] == "F" and (nextState in visitedFlags) == False):
        visitedFlags.append(nextState)
        return 0.5
    elif(maze[nextState[0]][nextState[1]] == "F" and (nextState in visitedFlags) == True):
        return -0.2
    baseReward = rewards[maze[nextState[0]][nextState[1]]]
    secReward = 0.05*(nextState[0] + nextState[1])
    return baseReward + secReward




alpha = 0.5
gamma = 0.7
episodes = 100
visitedFlags =[]
for episode in range(episodes):
    state = (0, 0)
    while maze[state[0]][state[1]] != 'T':
        action = select_action(state, QTable)
        nextState = move(state, action)
        reward = reward_calc(maze, nextState, rewards, visitedFlags)
        QTable[state[0], state[1], actions.index(action)] = (1 - alpha) * QTable[state[0], state[1], actions.index(action)]+ alpha * (reward + gamma * np.max(QTable[nextState[0], nextState[1]]) )
        state = nextState

state = (0, 0)
while maze[state[0], state[1]] != 'T':
    action = select_action_final(QTable, state)
    nextState = move(state, action)
    state = nextState
    print(state, end=' ')
