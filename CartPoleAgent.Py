import numpy as np

class Agent():

    def __init__(self, NumActions, Epsilon = 0.1, LearningRate = 1):
        self.NumActions = NumActions
        self.Epsilon = Epsilon 
        self.LearningRate = LearningRate
        self.Position = 0
        self.Velocity = 0
        self.Angle = 0
        self.AngleVel = 0

    def UpdateState(self, newState):
        self.Position = newState[0]
        self.Velocity = newState[1]
        self.Angle = newState[2]
        self.AngleVel = newState[3]

    def PickAction(self):
        if self.AngleVel < 0:
            return 0
        else:
            return 1

    def PickCurrentBest(self, state):
        if np.random.rand() > self.Epsilon:
            return np.argmax(self.QDict[state])
        else:
            return np.random.randint(0, self.NumActions)

    def UpdateEpsilon(self, amount=.9):
        self.Epsilon *= amount

    def PrintState(self):
        print(f"cart position: {self.Position}\ncart velocity: {self.Velocity}\ncart angle: {self.Angle}\ncart angular velocity: {self.AngleVel}")