import random
import math
import os

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

"""
q_table 缺点是动作没有连贯性，可能上次攻击了A，正在攻击时，就要攻击B
        如何优化？  army位置没有参考？ 我们有敌人的位置计算，分隔成4*4==16  还要计算自己主力位置
        系统可以自己便利是否要继续进攻。
        之前的策略都过于低级，需要更高级进攻策略
        1  集结部队    2  进攻敌人分矿   3  进攻敌人主基地   4  撤退
"""


_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_MOVE_MINIMAP = actions.FUNCTIONS.Move_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45 
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_NEUTRAL_MINERAL_FIELD = 341

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

DATA_FILE = 'my_refined_agent_data2'

ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_ATTACK = 'attack'

MAX_DEPOT_COUNT = 3

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MARINE,
    ACTION_ATTACK,
]

"""
for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 16) + '_' + str(mm_y - 16))
print ("------------------smart_actions start")
print (smart_actions)
print ("------------------smart_actions end")
"""
# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.disallowed_actions = {}

    def choose_action(self, observation, excluded_actions=[]):
        self.check_state_exist(observation)
        
        self.disallowed_actions[observation] = excluded_actions
        
        state_action = self.q_table.ix[observation, :]
        
        for excluded_action in excluded_actions:
            del state_action[excluded_action]

        if np.random.uniform() < self.epsilon:
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            
            action = state_action.idxmax()
        else:
            action = np.random.choice(state_action.index)
            
        return action

    def learn(self, s, a, r, s_):
        if s == s_:
            return
        
        self.check_state_exist(s_)
        self.check_state_exist(s)
        
        q_predict = self.q_table.ix[s, a]
        
        s_rewards = self.q_table.ix[s_, :]
        
        if s_ in self.disallowed_actions:
            for excluded_action in self.disallowed_actions[s_]:
                del s_rewards[excluded_action]
        
        if s_ != 'terminal':
            q_target = r + self.gamma * s_rewards.max()
        else:
            q_target = r  # next state is terminal
            
        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

class SparseAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SparseAgent, self).__init__()
        
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        
        self.previous_action = None
        self.previous_state = None
        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0
        
        self.cc_y = None
        self.cc_x = None
        self.attack_x = -1
        self.attack_y = -1
        self.searchcount = 0
        self.searchcurindex = 0
        self.move_number = 0
        self.cmdindex = 0
        
        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')
        
    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        
        return [x + x_distance, y + y_distance]
    
    def transformLocation(self, x, y):
        if not self.base_top_left:
            return [64 - x, 64 - y]
        
        return [x, y]

    # 合理的探测敌人,怎么算没有敌人呢？ 探测若干次?
    def getAttackPos(self ,supply_free) :
        if self.searchcount % 20 == 0:
            self.searchcurindex = random.randint(0, 3)
        i = self.searchcurindex
        if  self.base_top_left:
        #探测自己分矿
            if i == 0:
                return 15,47
        #探测敌人分矿
            if i == 1:
                return 47,15
        #探测敌人主矿
            if i ==2:
                if supply_free < 5:
                    return 15,15
                else:
                    return 31,31
        else: 
            if i == 0:
                return 47,15
            if i == 1:
                return 15,47
            if i == 2:
                if supply_free < 5:
                    return 47,47
                else:
                    return 31,31
        return 31,31
        
    def splitAction(self, action_id):
        smart_action = smart_actions[action_id]
            
        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return (smart_action, x, y)

    def CheckSaveAttack(self,x,y,friendly_y, friendly_x):
        for i in range(0, len(friendly_y)):
            if x == friendly_x[i] and y == friendly_y[i]:
                return 1
        return 0

    def step(self, obs):
        super(SparseAgent, self).step(obs)
        
        if obs.last():
            reward = obs.reward
        
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward*10, 'terminal')
            
            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
            
            self.previous_action = None
            self.previous_state = None
            
            self.move_number = 0
            
            return actions.FunctionCall(_NO_OP, [])
        
        unit_type = obs.observation['screen'][_UNIT_TYPE]

        if obs.first():
            self.attack_x = -1
            self.searchcount = 0
            self.cmdindex = 0
            player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
            if len(player_x) > 0:
                print ( "self.base_top_left %d %d %d" % (self.base_top_left, player_x.mean(),player_y.mean()) )
            else:
                print ( "self.base_top_left %d player_x==0" % (self.base_top_left) )

            self.cc_y, self.cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

        self.cmdindex = self.cmdindex + 1
        self.searchcount = self.searchcount + 1
        cc_y, cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
        cc_count = 1 if cc_y.any() else 0
        
        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        supply_depot_count = int(round(len(depot_y) / 69))

        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        barracks_count = int(round(len(barracks_y) / 137))
            
        supply_used = obs.observation['player'][3]
        supply_limit = obs.observation['player'][4]
        army_supply = obs.observation['player'][5]
        worker_supply = obs.observation['player'][6]

        killed_unit_score = obs.observation['score_cumulative'][5]
        killed_building_score = obs.observation['score_cumulative'][6]
        
        supply_free = supply_limit - supply_used
        
        if self.move_number == 0:
            self.move_number += 1
            
            current_state = np.zeros(13)
            current_state[0] = cc_count
            current_state[1] = supply_depot_count
            current_state[2] = barracks_count
            current_state[3] = obs.observation['player'][_ARMY_SUPPLY]
    
            hot_squares = np.zeros(4)        
            enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
            for i in range(0, len(enemy_y)):
                y = int(math.ceil((enemy_y[i] + 1) / 32))
                x = int(math.ceil((enemy_x[i] + 1) / 32))
                
                hot_squares[((y - 1) * 2) + (x - 1)] = 1
            #print ("enemy num %d" % (len(enemy_y)))
            #if not self.base_top_left:
            #   hot_squares = hot_squares[::-1]
            
            for i in range(0, 4):
                current_state[i + 4] = hot_squares[i]
            #
            green_squares = np.zeros(4)        
            friendly_y, friendly_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            #print ("friendly num %d" % (len(friendly_y)))
            for i in range(0, len(friendly_y)):
                y = int(math.ceil((friendly_y[i] + 1) / 32))
                x = int(math.ceil((friendly_x[i] + 1) / 32))
                
                green_squares[((y - 1) * 2) + (x - 1)] = 1
            
            #if not self.base_top_left:
            #    green_squares = green_squares[::-1]
            
            for i in range(0, 4):
                current_state[i + 8] = green_squares[i]

            if  self.base_top_left:
                current_state[12] = 1
            else:
                current_state[12] = 0
    
            if self.previous_action is not None:
                reward = 0                
                if killed_unit_score > self.previous_killed_unit_score:
                    reward += 0.1                    
                if killed_building_score > self.previous_killed_building_score:
                    reward += 0.2
                self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))
            # 奖励拆毁对方建筑 杀死对方人...
            self.previous_killed_unit_score = killed_unit_score
            self.previous_killed_building_score = killed_building_score
            
            excluded_actions = []
            if supply_depot_count == MAX_DEPOT_COUNT or worker_supply == 0:
                excluded_actions.append(1)
                
            if supply_depot_count == 0 or barracks_count == 2 or worker_supply == 0:
                excluded_actions.append(2)

            if supply_free == 0 or barracks_count == 0:
                excluded_actions.append(3)

            # 1234
            if army_supply == 0:
                excluded_actions.append(4)
                #excluded_actions.append(5)
                #excluded_actions.append(6)
                #excluded_actions.append(7)

            rl_action = self.qlearn.choose_action(str(current_state), excluded_actions)

            self.previous_state = current_state
            self.previous_action = rl_action
        
            smart_action, x, y = self.splitAction(self.previous_action)
            # 只有缺少建筑时采选SVCV，那么可能导致SCV不干活三
            if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

                if _SELECT_IDLE_WORKER in obs.observation['available_actions']:
                    print ("_SELECT_IDLE_WORKER")
                    return actions.FunctionCall(_SELECT_IDLE_WORKER, [_NOT_QUEUED])

                if unit_y.any():
                    i = random.randint(0, len(unit_y) - 1)
                    target = [unit_x[i], unit_y[i]]
                    #print ("_SELECT_POINT  _TERRAN_SCV")
                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
                
            elif smart_action == ACTION_BUILD_MARINE:
                if barracks_y.any():
                    #print ("_SELECT_POINT  barracks")
                    i = random.randint(0, len(barracks_y) - 1)
                    target = [barracks_x[i], barracks_y[i]]
            
                    return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])
                
            elif smart_action == ACTION_ATTACK:
                if _SELECT_ARMY in obs.observation['available_actions']:
                    #print ("_SELECT_ARMY")
                    return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
        
        elif self.move_number == 1:
            self.move_number += 1
            
            smart_action, x, y = self.splitAction(self.previous_action)
                
            if smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                if supply_depot_count < MAX_DEPOT_COUNT and _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                    if self.cc_y.any():
                        if supply_depot_count == 0:
                            target = self.transformDistance(round(self.cc_x.mean()), -35, round(self.cc_y.mean()), 0)
                        elif supply_depot_count == 1:
                            target = self.transformDistance(round(self.cc_x.mean()), -25, round(self.cc_y.mean()), -25)
                        elif supply_depot_count == 2:
                            target = self.transformDistance(round(self.cc_x.mean()), -30, round(self.cc_y.mean()), -15)
    
                        return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])
            
            if smart_action == ACTION_BUILD_BARRACKS:
                if barracks_count < 2 and _BUILD_BARRACKS in obs.observation['available_actions']:
                    if self.cc_y.any():
                        if  barracks_count == 0:
                            target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), -9)
                        elif  barracks_count == 1:
                            target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), 12)
    
                        return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])
            #if supply_free > 10 and smart_action == ACTION_ATTACK:
            #    smart_action = ACTION_BUILD_MARINE

            if smart_action == ACTION_BUILD_MARINE:
                if _TRAIN_MARINE in obs.observation['available_actions']:
                    return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
        
            elif smart_action == ACTION_ATTACK:
                do_it = True
                
                if len(obs.observation['single_select']) > 0 and obs.observation['single_select'][0][0] == _TERRAN_SCV:
                    do_it = False
                
                if len(obs.observation['multi_select']) > 0 and obs.observation['multi_select'][0][0] == _TERRAN_SCV:
                    do_it = False
                # 暴兵以后再rush
                #if supply_free>12:
                #    do_it = False

                enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
                targetx = -1
                targety = -1
                dislast = 9999
                for i in range(0, len(enemy_y)):
                    if  self.base_top_left:
                        #55  55
                        dis = abs(55-enemy_x[i]) + abs(55-enemy_y[i])
                        if dis < dislast:
                            targetx = enemy_x[i]
                            targety = enemy_y[i]
                    else:
                        #15 15
                        dis = abs(15-enemy_x[i]) + abs(15-enemy_y[i])
                        if dis < dislast:
                            targetx = enemy_x[i]
                            targety = enemy_y[i]
                
                if do_it and _ATTACK_MINIMAP in obs.observation["available_actions"]:
                    x_offset = random.randint(-1, 1)
                    y_offset = random.randint(-1, 1)
                    friendly_y, friendly_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
                    if targetx <10000 : # == -1:  # 放弃谈心算法  是不是敌人有影子？
                        x,y = self.getAttackPos(supply_free)
                        #print ("_ATTACK_MINIMAP army random cmdindex %d %d %d " % (self.cmdindex,int(x),int(y)))
                        tx = int(x) + (x_offset * 8)
                        ty = int(y) + (y_offset * 8)
                        if self.CheckSaveAttack(tx,ty,friendly_y, friendly_x) == 0 and targetx != -1:
                            return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, self.transformLocation(int(x) + (x_offset * 8), int(y) + (y_offset * 8))])
                        else:
                            return actions.FunctionCall(_MOVE_MINIMAP, [_NOT_QUEUED, self.transformLocation(int(x) + (x_offset * 8), int(y) + (y_offset * 8))])
                    else:
                        #print ("_ATTACK_MINIMAP army greedy cmdindex %d %d %d" % (self.cmdindex,targetx,targety))
                        if self.CheckSaveAttack(targetx,targety,friendly_y, friendly_x) == 0 :
                            return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, self.transformLocation(targetx, targety )])
                        else:
                            return actions.FunctionCall(_MOVE_MINIMAP, [_NOT_QUEUED, self.transformLocation(targetx, targety )])
        elif self.move_number == 2:
            self.move_number = 0
            
            smart_action, x, y = self.splitAction(self.previous_action)
                
            if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                if _HARVEST_GATHER in obs.observation['available_actions']:
                    unit_y, unit_x = (unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()
                    
                    if unit_y.any():
                        i = random.randint(0, len(unit_y) - 1)
                        
                        m_x = unit_x[i]
                        m_y = unit_y[i]
                        
                        target = [int(m_x), int(m_y)]
                        
                        return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])
        
        return actions.FunctionCall(_NO_OP, [])
