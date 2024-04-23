    # def PickWithFutureBest(self, state):
    #     if np.random.rand() > self.Epsilon:
    #         return self.BestPathValue(state, 0)
    #     else:
    #         return np.random.randint(0, self.NumActions)
        
    # def BestPathValue(self, state, currentDepth):
    #     if np.max(self.QDict[state]) > 1000:
    #         return np.max(self.QDict[state])
    #     if currentDepth >= 3:
    #         return np.max(self.QDict[state])
    #     actionsToChoose = []
    #     for actionOption in range(self.QDict[state].size):
    #         newLocation = self.simluateNewState(state, actionOption)
    #         actionsToChoose.append(self.QDict[state] + self.BestPathValue(newLocation, currentDepth+1))
    #     return np.argmax(actionsToChoose)

    # def simluateNewState(self, currentState, action):
    #     #0 is left, 1 is down, 2 is right, 3 is up
    #     #on slip, it's 1/3 to go left of selected, 1/3 to go right of selected, 1/3 to go selected
    #     slip = np.random.randint(0,3)
    #     actionSimulated = action
    #     match action:
    #         case 0: 
    #             match slip:
    #                 case 1:
    #                     actionSimulated = 3
    #                 case 2:
    #                     actionSimulated = 1
    #         case 1: 
    #             match slip:
    #                 case 1:
    #                     actionSimulated = 2
    #                 case 2:
    #                     actionSimulated = 0
    #         case 2: 
    #             match slip:
    #                 case 1:
    #                     actionSimulated = 1
    #                 case 2:
    #                     actionSimulated = 3
    #         case 3: 
    #             match slip:
    #                 case 1:
    #                     actionSimulated = 2
    #                 case 2:
    #                     actionSimulated = 0


    #     match actionSimulated:
    #         case 0:
    #             if currentState == 0 or currentState % self.MapSize == 0:
    #                 newState = currentState
    #             else:
    #                 newState = currentState - 1
    #         case 1:
    #             if currentState >= (self.MapSize*(self.MapSize-1)) and currentState < (self.NumStates):
    #                 newState = currentState
    #             else:
    #                 newState = currentState + self.MapSize
    #         case 2:
    #             if currentState == self.MapSize - 1 or (currentState + 1) % self.MapSize == 0:
    #                 newState = currentState
    #             else:
    #                 newState = currentState + 1
    #         case 3:
    #             if currentState >= 0 and currentState < self.MapSize:
    #                 newState = currentState
    #             else:
    #                 newState = currentState - self.MapSize
    #     return newState

def CarefulPolicy(state):
        moves = {'<':0,'V':1,'>':2,'^':3} 
        board = ['<','^','V','^'
                ,'<','_','>','_'
                ,'^','V','<','_'
                ,'_','>','>','']
        return moves[board[state]]
    
def StraightPolicy(state):
        moves = {'<':0,'V':1,'>':2,'^':3} 
        board = ['V','>','V','<'
                ,'V','_','V','_'
                ,'>','V','V','_'
                ,'_','>','>','']
        return moves[board[state]]