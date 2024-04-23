import numpy as np

class Agent():

    def __init__(self, NumActions, MapSize = 4, Epsilon = 0.1, LearningRate = 1):
        self.NumActions = NumActions
        self.MapSize = MapSize
        self.NumStates = MapSize*MapSize
        self.Epsilon = Epsilon 
        self.QDict = np.zeros((self.NumStates, NumActions))
        self.ValDict = np.zeros(self.NumStates)
        self.LearningRate = LearningRate

    def PickCurrentBest(self, state):
        if np.random.rand() > self.Epsilon:
            return np.argmax(self.QDict[state])
        else:
            return np.random.randint(0, self.NumActions)
        
    def UpdateQDictValue(self, reward, state, action, next_state, gamma=.5):
        QVal = reward + gamma * self.QDict[next_state][self.PickCurrentBest(next_state)]
        self.QDict[state][action] += (QVal - self.QDict[state][action]) * self.LearningRate

    def PrintQDict(self):
        outString = ""
        for x in range(self.NumStates):
            maxVal = round(np.max(self.QDict[x]),2)
            outString += f"[{maxVal:0.2f}], "
            if (x+1) % self.MapSize == 0:
                outString += "\n"
        print(outString)

    def UpdateEpsilon(self, amount=.9):
        self.Epsilon *= amount