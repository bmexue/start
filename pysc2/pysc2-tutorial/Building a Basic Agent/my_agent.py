from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import time

# Functions
_BUILD_REFINERY = actions.FUNCTIONS.Build_Refinery_screen.id
_BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_COMMANDCENTER = actions.FUNCTIONS.Build_CommandCenter_screen.id
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id
_NOOP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Unit IDs
_TERRAN_COMMANDCENTER = 18
_VESPENE_GAS = 342
_TERRAN_SUPPLYDEPOT = 19
_TERRAN_SCV = 45
_SUPPLY_USED = 3
_SUPPLY_MAX = 4

# Parameters
_PLAYER_SELF = 1
_NOT_QUEUED = [0]
_QUEUED = [1]


##x=-1
##y=-1

class SimpleAgent(base_agent.BaseAgent):
    base_top_left = None
    supply_depot_built = False
    commandcenters_selected = False
    commandcenter_built = False
    scv_selected = False
    refinery_built = False

    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        
        return [x + x_distance, y + y_distance]
    
    def step(self, obs):
        super(SimpleAgent, self).step(obs)
        
        time.sleep(0.5)
        
        if self.base_top_left is None:
            player_y, player_x = (obs.observation["minimap"][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = player_y.mean() <= 31
            
        if not self.commandcenter_built:
            if not self.scv_selected:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
                target = [unit_x[0], unit_y[0]]
                self.scv_selected = True
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
            elif _BUILD_COMMANDCENTER in obs.observation["available_actions"]:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
                target = self.transformLocation(int(unit_x.mean()), 20, int(unit_y.mean()), 0)
                print(unit_x.mean(), unit_y.mean())
                self.commandcenter_built = True
                return actions.FunctionCall(_BUILD_COMMANDCENTER, [_NOT_QUEUED, target])
        elif not self.commandcenters_selected:
            unit_type = obs.observation["screen"][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
            for i in range(len(unit_y)):
                target = [int(unit_x[0].mean()), int(unit_y[0].mean())]
                return actions.FunctionCall(_SELECT_POINT, [_QUEUED, target])
            self.commandcenters_selected = True
        elif obs.observation["player"][_SUPPLY_USED] < obs.observation["player"][_SUPPLY_MAX] and _TRAIN_SCV in obs.observation["available_actions"]:
            print("building scvs")
            return actions.FunctionCall(_TRAIN_SCV, [_QUEUED])
##        elif not self.supply_depot_built:
##            if not self.scv_selected:
##                unit_type = obs.observation["screen"][_UNIT_TYPE]
##                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
##                target = [unit_x[0], unit_y[0]]
##                self.scv_selected = True
##                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
##            elif _BUILD_SUPPLYDEPOT in obs.observation["available_actions"]:
##                unit_type = obs.observation["screen"][_UNIT_TYPE]
##                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
##                target = self.transformLocation(int(unit_x.mean()), 20, int(unit_y.mean()), 0)
##                print(unit_x.mean(), unit_y.mean())
##                self.supply_depot_built = True
##                return actions.FunctionCall(_BUILD_SUPPLYDEPOT, [_NOT_QUEUED, target])
##        elif not self.refinery_built:
##            if not self.scv_selected:
##                unit_type = obs.observation["screen"][_UNIT_TYPE]
##                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
##                target = [unit_x[0], unit_y[0]]
##                self.scv_selected = True
##                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
##            if _BUILD_REFINERY in obs.observation["available_actions"]:
##                unit_type = obs.observation["screen"][_UNIT_TYPE]
##                unit_y, unit_x = (unit_type == _VESPENE_GAS).nonzero()
##                target = [unit_x[0].mean(), unit_y[0].mean()]
##                self.refinery_built = True
##                return actions.FunctionCall(_BUILD_REFINERY, [_QUEUED, target])
                    
        return actions.FunctionCall(_NOOP, [])