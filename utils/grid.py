import sys

class Grid:
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        self.states = rows*columns
        self.joint_states = self.states*self.states


    def joint_state_to_individual_states(self, joint_state):
        '''Converts a joint state to the individual states.'''
        state1 = int(joint_state/self.states)
        state2 = joint_state-state1*self.states
        return state1,state2


    def individual_to_joint_states(self, state1, state2):
        '''Converts the individual states to a joint state.'''
        return state1*self.states+state2

    def next_state(self, state, action):
        '''Returns the next individual state given an individual state and an action.'''
        row =  int(state/self.columns)
        col = state-row*self.columns
        if action == 0: # up
            row = row-1
        elif action == 1: # right
            col = col+1
        elif action == 2: # down
            row = row+1
        elif action == 3: # left
            col = col-1
        else:
            print('Invalid action')
            sys.exit(-1)
        row = max(0,row)
        row = min(row, self.rows-1)
        col = max(0,col)
        col = min(col, self.columns-1)
        return row*self.columns+col