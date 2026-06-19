import random
from datetime import datetime
import time

class Simulation:
    '''
        This class contains the implementation of the simulation system.
        It initializes the simulation with a given state and provides methods to run the simulation.
    '''
    def __init__(self,state):
        self.filling_tank_time = 1e6
        self.emptying_tank_time = 1e6

        self.level_B201 = state['level_B201']
        self.level_B202 = state['level_B202']
        self.level_B203 = state['level_B203']
        self.level_B204 = state['level_B204']

        self.max_height_B201 = state['max_height_B201']
        self.min_height_B201 = state['min_height_B201']
        self.max_height_B202 = state['max_height_B202']
        self.min_height_B202 = state['min_height_B202']
        self.max_height_B203 = state['max_height_B203']
        self.min_height_B203 = state['min_height_B203']
        self.max_height_B204 = state['max_height_B204']
        self.min_height_B204 = state['min_height_B204']
        
        self.time = 0
        now = datetime.now()
        self.timer = now.strftime("%Y-%m-%d %H:%M:%S")
        self.max_time = state['time']
        self.valve_in_B201 = state['valve_in_B201']
        self.valve_in_B202 = state['valve_in_B202']
        self.valve_in_B203 = state['valve_in_B203']
        self.valve_in_B204 = state['valve_in_B204']
        self.valve_out_B201 = state['valve_out_B201']
        self.valve_out_B202 = state['valve_out_B202']
        self.valve_out_B203 = state['valve_out_B203']
        self.valve_out_B204 = state['valve_out_B204']

        self.tank_B201_state = state['tank_B201_state']
        self.tank_B202_state = state['tank_B202_state']
        self.tank_B203_state = state['tank_B203_state']
        self.tank_B204_state = state['tank_B204_state']
        
        self.condition = state['condition']

        self.pump_alt = state['pump_alt']
        self.pump_main = state['pump_main']

        self.pump_power = state['pump_power']
        self.pump_power_alt = state['pump_power_alt']

        self.log = {'time':[],
                    'level_B201':[],
                    'level_B202':[],
                    'level_B203':[],
                    'level_B204':[],
                    'valve_in_B201':[],
                    'valve_in_B202':[],
                    'valve_in_B203':[],
                    'valve_in_B204':[],
                    'valve_out_B201':[],
                    'valve_out_B202':[],
                    'valve_out_B203':[],
                    'valve_out_B204':[],
                    'volumetric_flow_rate':[],
                    'pump_alt':[],
                    'pump_main':[],
                    'state_filling_tank_B201':[],
                    'state_filling_tank_B202':[],
                    'state_filling_tank_B203':[],
                    'state_filling_tank_B204':[],
                    'state_emptying_tank_B201':[],
                    'state_emptying_tank_B202':[],
                    'state_emptying_tank_B203':[],
                    'state_emptying_tank_B204':[],
                    'sensor_continuous_tank':[],
                    'sensor_discrete_tank_B201_low':[],
                    'sensor_discrete_tank_B201_medium':[],
                    'sensor_discrete_tank_B201_high':[],
                    'sensor_discrete_tank_B202_low':[],
                    'sensor_discrete_tank_B202_medium':[],
                    'sensor_discrete_tank_B202_high':[],
                    'sensor_discrete_tank_B203_low':[],
                    'sensor_discrete_tank_B203_medium':[],
                    'sensor_discrete_tank_B203_high':[],
                    'sensor_discrete_tank_B204_low':[],
                    'sensor_discrete_tank_B204_medium':[],
                    'sensor_discrete_tank_B204_high':[],
                    'condition':[]}
    
        
    def get_volumetric_flow_rate(self):
        if self.pump_power > 0.1 or self.pump_power_alt > 0.1:
            if len(self.log['level_B204']) < 2:
                self.log['volumetric_flow_rate'].append(0)
            else:
                flow_rate = self.log['level_B204'][-1] - self.log['level_B204'][-2]
                self.log['volumetric_flow_rate'].append(flow_rate)
        else:
            self.log['volumetric_flow_rate'].append(0)

    def get_state(self,flag):
        self.log['state_filling_tank_B201'].append(0)
        self.log['state_filling_tank_B202'].append(0)
        self.log['state_filling_tank_B203'].append(0)
        self.log['state_filling_tank_B204'].append(0)
        self.log['state_emptying_tank_B201'].append(0)
        self.log['state_emptying_tank_B202'].append(0)
        self.log['state_emptying_tank_B203'].append(0)
        self.log['state_emptying_tank_B204'].append(0)
        if flag=='state_filling_tank_B201':
            self.log['state_filling_tank_B201'][-1]=1
        if flag=='state_filling_tank_B202':
            self.log['state_filling_tank_B202'][-1]=1
        if flag=='state_filling_tank_B203':
            self.log['state_filling_tank_B203'][-1]=1
        if flag=='state_filling_tank_B204':
            self.log['state_filling_tank_B204'][-1]=1
        if flag=='state_emptying_tank_B201':
            self.log['state_emptying_tank_B201'][-1]=1
        if flag=='state_emptying_tank_B202':
            self.log['state_emptying_tank_B202'][-1]=1
        if flag=='state_emptying_tank_B203':
            self.log['state_emptying_tank_B203'][-1]=1
        if flag=='state_emptying_tank_B204':
            self.log['state_emptying_tank_B204'][-1]=1
        
    def get_time(self):
        times = self.timer
        sec = int(self.timer[-2:])+random.randint(1,5)
        min = int(self.timer[-5:-3])
        hr = int(self.timer[-8:-6])
        if sec >= 60:
            sec = sec - 60
            min = int(times[-5:-3])+1
            if min >= 60:
                min = min - 60
                hr = int(times[-8:-6])+1
                if hr >= 24:
                    hr = hr - 24
            else:
                hr = int(times[-8:-6])
        self.timer = times[:-8]+f"{hr:02d}"+":"+f"{min:02d}"+":"+f"{sec:02d}"
        return self.timer
    
    def check_level(self):
        self.log['sensor_discrete_tank_B201_low'].append(0)
        self.log['sensor_discrete_tank_B201_medium'].append(0)
        self.log['sensor_discrete_tank_B201_high'].append(0)
        self.log['sensor_discrete_tank_B202_low'].append(0)
        self.log['sensor_discrete_tank_B202_medium'].append(0)
        self.log['sensor_discrete_tank_B202_high'].append(0)
        self.log['sensor_discrete_tank_B203_low'].append(0)
        self.log['sensor_discrete_tank_B203_medium'].append(0)
        self.log['sensor_discrete_tank_B203_high'].append(0)
        self.log['sensor_discrete_tank_B204_low'].append(0)
        self.log['sensor_discrete_tank_B204_medium'].append(0)
        self.log['sensor_discrete_tank_B204_high'].append(0)
        if self.level_B201 > 0.0:
            self.log['sensor_discrete_tank_B201_low'][-1]=1
        if self.level_B201 > 0.024:
            self.log['sensor_discrete_tank_B201_medium'][-1]=1
        if self.level_B201 > 0.03:
            self.log['sensor_discrete_tank_B201_high'][-1]=1
        
        if self.level_B202 > 0.0:
            self.log['sensor_discrete_tank_B202_low'][-1]=1
        if self.level_B202 > 0.024:
            self.log['sensor_discrete_tank_B202_medium'][-1]=1
        if self.level_B202 > 0.03:
            self.log['sensor_discrete_tank_B202_high'][-1]=1

        if self.level_B203 > 0.0:
            self.log['sensor_discrete_tank_B203_low'][-1]=1
        if self.level_B203 > 0.024:
            self.log['sensor_discrete_tank_B203_medium'][-1]=1
        if self.level_B203 > 0.03:
            self.log['sensor_discrete_tank_B203_high'][-1]=1
        
        if self.level_B204 > 0.0:
            self.log['sensor_discrete_tank_B204_low'][-1]=1
        if self.level_B204 > 0.024:
            self.log['sensor_discrete_tank_B204_medium'][-1]=1
        if self.level_B204 > 0.05:
            self.log['sensor_discrete_tank_B204_high'][-1]=1


    def state_filling_tank_B201(self):
        '''
            This method updates the state of tank B201 based on the current state and the valve settings.
        '''
        time_limit = int(min(self.max_time-self.time, self.filling_tank_time))
        print('time_limit:', time_limit)
        for i in range(time_limit):
            self.time +=1/10
            if self.valve_in_B201:
                self.level_B201 += 0.001 + random.uniform(-0.0002, 0.0002)  # Adding some randomness to the filling rate
                self.log['time'].append(self.get_time())
                self.log['level_B201'].append(self.level_B201)
                self.log['level_B202'].append(self.level_B202)
                self.log['level_B203'].append(self.level_B203)
                self.log['level_B204'].append(self.level_B204)
                self.log['valve_in_B201'].append(self.valve_in_B201)
                self.log['valve_in_B202'].append(self.valve_in_B202)
                self.log['valve_in_B203'].append(self.valve_in_B203)
                self.log['valve_in_B204'].append(self.valve_in_B204)
                self.log['valve_out_B201'].append(self.valve_out_B201)
                self.log['valve_out_B202'].append(self.valve_out_B202)
                self.log['valve_out_B203'].append(self.valve_out_B203)
                self.log['valve_out_B204'].append(self.valve_out_B204)
                self.log['pump_alt'].append(self.pump_alt*self.pump_power_alt)
                self.log['pump_main'].append(self.pump_main*self.pump_power)
                self.log['condition'].append(self.condition)
                self.check_level()
                self.get_state('state_filling_tank_B201')
                self.get_volumetric_flow_rate()
                self.log['sensor_continuous_tank'].append(0)
                if self.level_B201 >= self.max_height_B201:
                    # print('Tank B201 is full.')
                    self.valve_in_B201 = 0
                    self.log['valve_in_B201'][-1] = self.valve_in_B201
                    return self.log
            else:
                print('Error')
                raise ValueError("Valve in B201 is closed, cannot fill the tankB201.")
        
    def state_filling_tank_B202(self):
        '''
            This method updates the state of tank B202 based on the current state and the valve settings.
        '''
        time_limit = int(min(self.max_time-self.time, self.filling_tank_time))
        print('time_limit:', time_limit)
        for i in range(time_limit):
            self.time +=1/10
            if self.valve_in_B202:
                self.level_B202 += 0.001 + random.uniform(-0.0002, 0.0002)
                self.log['time'].append(self.get_time())
                self.log['level_B201'].append(self.level_B201)
                self.log['level_B202'].append(self.level_B202)
                self.log['level_B203'].append(self.level_B203)
                self.log['level_B204'].append(self.level_B204)
                self.log['valve_in_B201'].append(self.valve_in_B201)
                self.log['valve_in_B202'].append(self.valve_in_B202)
                self.log['valve_in_B203'].append(self.valve_in_B203)
                self.log['valve_in_B204'].append(self.valve_in_B204)
                self.log['valve_out_B201'].append(self.valve_out_B201)
                self.log['valve_out_B202'].append(self.valve_out_B202)
                self.log['valve_out_B203'].append(self.valve_out_B203)
                self.log['valve_out_B204'].append(self.valve_out_B204)
                self.log['pump_alt'].append(self.pump_alt*self.pump_power_alt)
                self.log['pump_main'].append(self.pump_main*self.pump_power)
                self.log['condition'].append(self.condition)
                self.check_level()
                self.get_state('state_filling_tank_B202')
                self.get_volumetric_flow_rate()
                self.log['sensor_continuous_tank'].append(0)
                if self.level_B202 >= self.max_height_B202:
                    print('Tank B202 is full.')
                    self.valve_in_B202 = 0
                    self.log['valve_in_B202'][-1] = self.valve_in_B202
                    return self.log
            else:
                print('Error')
                raise ValueError("Valve in B202 is closed, cannot fill the tankB202.")
        
    
    def state_filling_tank_B203(self):
        '''
            This method updates the state of tank B203 based on the current state and the valve settings.
        '''
        time_limit = int(min(self.max_time-self.time, self.filling_tank_time))
        print('time_limit:', time_limit)
        for i in range(time_limit):
            self.time +=1/10
            if self.valve_in_B203:
                self.level_B203 += 0.001 + random.uniform(-0.0002, 0.0002)
                self.log['time'].append(self.get_time())
                self.log['level_B201'].append(self.level_B201)
                self.log['level_B202'].append(self.level_B202)
                self.log['level_B203'].append(self.level_B203)
                self.log['level_B204'].append(self.level_B204)
                self.log['valve_in_B201'].append(self.valve_in_B201)
                self.log['valve_in_B202'].append(self.valve_in_B202)
                self.log['valve_in_B203'].append(self.valve_in_B203)
                self.log['valve_in_B204'].append(self.valve_in_B204)
                self.log['valve_out_B201'].append(self.valve_out_B201)
                self.log['valve_out_B202'].append(self.valve_out_B202)
                self.log['valve_out_B203'].append(self.valve_out_B203)
                self.log['valve_out_B204'].append(self.valve_out_B204)
                self.log['pump_alt'].append(self.pump_alt*self.pump_power_alt)
                self.log['pump_main'].append(self.pump_main*self.pump_power)
                self.log['condition'].append(self.condition)
                self.check_level()
                self.get_volumetric_flow_rate()
                self.get_state('state_filling_tank_B203')
                self.log['sensor_continuous_tank'].append(0)
                if self.level_B203 >= self.max_height_B203:
                    print('Tank B203 is full.')
                    self.valve_in_B203 = 0
                    self.log['valve_in_B203'][-1] = self.valve_in_B203
                    return self.log
            else:
                print('Error')
                raise ValueError("Valve in B203 is closed, cannot fill the tankB203.")
        
    
    def state_emptying_tank_B201(self):
        '''
            This method updates the state of tank B201 and B204 based on the current state and the valve settings.
        '''
        time_limit = int(min(self.max_time-self.time, self.filling_tank_time))
        print('time_limit:', time_limit)
        while self.level_B201>self.min_height_B201 and time_limit>0:
            self.time +=1/10
            noise = +random.uniform(-0.00002, 0.00002)
            if self.valve_out_B201 and self.valve_in_B204 and self.pump_power:
                water_pumped = min(0.00001*self.pump_power*self.pump_main+noise,self.level_B201)
                self.level_B201 -= water_pumped
                if self.condition == 'leak':
                    water_pumped = water_pumped * random.uniform(0.7,0.9)
                self.level_B204 += water_pumped
                self.log['time'].append(self.get_time())
                self.log['level_B201'].append(self.level_B201)
                self.log['level_B202'].append(self.level_B202)
                self.log['level_B203'].append(self.level_B203)
                self.log['level_B204'].append(self.level_B204)
                self.log['valve_in_B201'].append(self.valve_in_B201)
                self.log['valve_in_B202'].append(self.valve_in_B202)
                self.log['valve_in_B203'].append(self.valve_in_B203)
                self.log['valve_in_B204'].append(self.valve_in_B204)
                self.log['valve_out_B201'].append(self.valve_out_B201)
                self.log['valve_out_B202'].append(self.valve_out_B202)
                self.log['valve_out_B203'].append(self.valve_out_B203)
                self.log['valve_out_B204'].append(self.valve_out_B204)
                self.log['pump_alt'].append(self.pump_alt*self.pump_power_alt)
                self.log['pump_main'].append(self.pump_main*self.pump_power)
                self.log['condition'].append(self.condition)
                self.check_level()
                self.get_state('state_emptying_tank_B201')
                self.get_volumetric_flow_rate()

                if self.condition == 'sensor_fault':
                    self.log['sensor_continuous_tank'].append(water_pumped*(random.uniform(0.8,0.9)))
                else:
                    self.log['sensor_continuous_tank'].append(water_pumped)
                if self.level_B201 <= self.min_height_B201:
                    print('Tank B201 is empty.')
                    self.valve_out_B201 = 0
                    self.log['valve_out_B201'][-1] = self.valve_out_B201
                    return self.log
                
            elif self.valve_out_B201 and self.valve_in_B204 and self.pump_power_alt:
                water_pumped = min(0.00001*self.pump_power_alt*self.pump_alt+noise,self.level_B201)
                self.level_B201 -= water_pumped
                if self.condition == 'leak':
                    water_pumped = water_pumped * random.uniform(0.7,0.9)
                self.level_B204 += water_pumped
                self.log['time'].append(self.get_time())
                self.log['level_B201'].append(self.level_B201)
                self.log['level_B202'].append(self.level_B202)
                self.log['level_B203'].append(self.level_B203)
                self.log['level_B204'].append(self.level_B204)
                self.log['valve_in_B201'].append(self.valve_in_B201)
                self.log['valve_in_B202'].append(self.valve_in_B202)
                self.log['valve_in_B203'].append(self.valve_in_B203)
                self.log['valve_in_B204'].append(self.valve_in_B204)
                self.log['valve_out_B201'].append(self.valve_out_B201)
                self.log['valve_out_B202'].append(self.valve_out_B202)
                self.log['valve_out_B203'].append(self.valve_out_B203)
                self.log['valve_out_B204'].append(self.valve_out_B204)
                self.log['pump_alt'].append(self.pump_alt*self.pump_power_alt)
                self.log['pump_main'].append(self.pump_main*self.pump_power)
                self.log['condition'].append(self.condition)

                self.check_level()
                self.get_volumetric_flow_rate()
                self.get_state('state_emptying_tank_B201')
                if self.condition == 'sensor_fault':
                    self.log['sensor_continuous_tank'].append(water_pumped*(random.uniform(0.8,0.9)))
                else:
                    self.log['sensor_continuous_tank'].append(water_pumped)
                if self.level_B201 <= self.min_height_B201:
                    print('Tank B201 is empty.')
                    self.valve_out_B201 = 0
                    self.log['valve_out_B201'][-1] = self.valve_out_B201
                    return self.log
            else:
                print('Error')
                valves={'valve_in_B204':self.valve_in_B204,'valve_out_B201':self.valve_out_B201,'pump_main':self.pump_main,'pump_alt':self.pump_alt}
                error_valves = [i for i in valves if valves[i]!=True]
                raise ValueError(f"{error_valves} is closed, cannot empty the tankB201.")

    def state_emptying_tank_B202(self):
        '''
            This method updates the state of tank B202 and B204 based on the current state and the valve settings.
        '''
        time_limit = int(min(self.max_time-self.time, self.filling_tank_time))
        print('time_limit:', time_limit)
        noise = +random.uniform(-0.00002, 0.00002)
        while self.level_B202>self.min_height_B202 and time_limit>0:
            self.time +=1/10
            if self.valve_out_B202 and self.valve_in_B204 and self.pump_power:
                water_pumped = min(0.00001*self.pump_power*self.pump_main+noise,self.level_B202)
                self.level_B202 -= water_pumped
                if self.condition == 'leak':
                    water_pumped = water_pumped * random.uniform(0.7,0.9)
                self.level_B204 += water_pumped
                self.log['time'].append(self.get_time())
                self.log['level_B201'].append(self.level_B201)
                self.log['level_B202'].append(self.level_B202)
                self.log['level_B203'].append(self.level_B203)
                self.log['level_B204'].append(self.level_B204)
                self.log['valve_in_B201'].append(self.valve_in_B201)
                self.log['valve_in_B202'].append(self.valve_in_B202)
                self.log['valve_in_B203'].append(self.valve_in_B203)
                self.log['valve_in_B204'].append(self.valve_in_B204)
                self.log['valve_out_B201'].append(self.valve_out_B201)
                self.log['valve_out_B202'].append(self.valve_out_B202)
                self.log['valve_out_B203'].append(self.valve_out_B203)
                self.log['valve_out_B204'].append(self.valve_out_B204)
                self.log['pump_alt'].append(self.pump_alt*self.pump_power_alt)
                self.log['pump_main'].append(self.pump_main*self.pump_power)
                self.log['condition'].append(self.condition)

                self.check_level()
                self.get_volumetric_flow_rate()
                self.get_state('state_emptying_tank_B202')
                if self.condition == 'sensor_fault':
                    self.log['sensor_continuous_tank'].append(water_pumped*(random.uniform(0.8,0.9)))
                else:
                    self.log['sensor_continuous_tank'].append(water_pumped)
                if self.level_B202 <= self.min_height_B202:
                    print('Tank B202 is empty.')
                    self.valve_out_B202 = 0
                    self.log['valve_out_B202'][-1] = self.valve_out_B202
                    return self.log
                
            elif self.valve_out_B202 and self.valve_in_B204 and self.pump_power_alt:
                water_pumped = min(0.00001*self.pump_power_alt*self.pump_alt+noise,self.level_B202)
                self.level_B202 -= water_pumped
                if self.condition == 'leak':
                    water_pumped = water_pumped * random.uniform(0.7,0.9)
                self.level_B204 += water_pumped
                self.log['time'].append(self.get_time())
                self.log['level_B201'].append(self.level_B201)
                self.log['level_B202'].append(self.level_B202)
                self.log['level_B203'].append(self.level_B203)
                self.log['level_B204'].append(self.level_B204)
                self.log['valve_in_B201'].append(self.valve_in_B201)
                self.log['valve_in_B202'].append(self.valve_in_B202)
                self.log['valve_in_B203'].append(self.valve_in_B203)
                self.log['valve_in_B204'].append(self.valve_in_B204)
                self.log['valve_out_B201'].append(self.valve_out_B201)
                self.log['valve_out_B202'].append(self.valve_out_B202)
                self.log['valve_out_B203'].append(self.valve_out_B203)
                self.log['valve_out_B204'].append(self.valve_out_B204)
                self.log['pump_alt'].append(self.pump_alt*self.pump_power_alt)
                self.log['pump_main'].append(self.pump_main*self.pump_power)
                self.log['condition'].append(self.condition)
                self.check_level()
                self.get_state('state_emptying_tank_B202')
                self.get_volumetric_flow_rate()
                if self.condition == 'sensor_fault':
                    self.log['sensor_continuous_tank'].append(water_pumped*(random.uniform(0.8,0.9)))
                else:
                    self.log['sensor_continuous_tank'].append(water_pumped)
                if self.level_B202 <= self.min_height_B202:
                    print('Tank B202 is empty.')
                    self.valve_out_B202 = 0
                    self.log['valve_out_B202'][-1] = self.valve_out_B202
                    return self.log
            else:
                print('Error')
                valves={'valve_in_B204':self.valve_in_B204,'valve_out_B201':self.valve_out_B201,'pump_main':self.pump_main,'pump_alt':self.pump_alt}
                error_valves = [i for i in valves if valves[i]!=True]
                raise ValueError(f"{error_valves} is closed, cannot empty the tankB202.")

    def state_emptying_tank_B203(self):
        '''
            This method updates the state of tank B203 and B204 based on the current state and the valve settings.
        '''
        time_limit = int(min(self.max_time-self.time, self.filling_tank_time))
        print('time_limit:', time_limit)
        noise = +random.uniform(-0.00002, 0.00002)
        while self.level_B203>self.min_height_B203 and time_limit>0:
            self.time +=1/10
            if self.valve_out_B203 and self.valve_in_B204 and self.pump_power:
                water_pumped = min(0.00001*self.pump_power*self.pump_main+noise,self.level_B203)
                self.level_B203 -= water_pumped
                if self.condition == 'leak':
                    water_pumped = water_pumped * random.uniform(0.7,0.9)
                self.level_B204 += water_pumped
                self.log['time'].append(self.get_time())
                self.log['level_B201'].append(self.level_B201)
                self.log['level_B202'].append(self.level_B202)
                self.log['level_B203'].append(self.level_B203)
                self.log['level_B204'].append(self.level_B204)
                self.log['valve_in_B201'].append(self.valve_in_B201)
                self.log['valve_in_B202'].append(self.valve_in_B202)
                self.log['valve_in_B203'].append(self.valve_in_B203)
                self.log['valve_in_B204'].append(self.valve_in_B204)
                self.log['valve_out_B201'].append(self.valve_out_B201)
                self.log['valve_out_B202'].append(self.valve_out_B202)
                self.log['valve_out_B203'].append(self.valve_out_B203)
                self.log['valve_out_B204'].append(self.valve_out_B204)
                self.log['pump_alt'].append(self.pump_alt*self.pump_power_alt)
                self.log['pump_main'].append(self.pump_main*self.pump_power)
                self.log['condition'].append(self.condition)
                self.check_level()
                self.get_state('state_emptying_tank_B203')
                self.get_volumetric_flow_rate()
                if self.condition == 'sensor_fault':
                    self.log['sensor_continuous_tank'].append(water_pumped*(random.uniform(0.8,0.9)))
                else:
                    self.log['sensor_continuous_tank'].append(water_pumped)
                if self.level_B203 <= self.min_height_B203:
                    print('Tank B203 is empty.')
                    # self.valve_out_B203 = 0
                    # self.pump_power = 0
                    self.log['valve_out_B203'][-1] = self.valve_out_B203
                    return self.log
                
            elif self.valve_out_B203 and self.valve_in_B204 and self.pump_power_alt:
                water_pumped=min(0.00001*self.pump_power_alt*self.pump_alt+noise,self.level_B203)
                self.level_B203 -= water_pumped
                if self.condition == 'leak':
                    water_pumped = water_pumped * random.uniform(0.7,0.9)
                self.level_B204 += water_pumped
                self.log['time'].append(self.get_time())
                self.log['level_B201'].append(self.level_B201)
                self.log['level_B202'].append(self.level_B202)
                self.log['level_B203'].append(self.level_B203)
                self.log['level_B204'].append(self.level_B204)
                self.log['valve_in_B201'].append(self.valve_in_B201)
                self.log['valve_in_B202'].append(self.valve_in_B202)
                self.log['valve_in_B203'].append(self.valve_in_B203)
                self.log['valve_in_B204'].append(self.valve_in_B204)
                self.log['valve_out_B201'].append(self.valve_out_B201)
                self.log['valve_out_B202'].append(self.valve_out_B202)
                self.log['valve_out_B203'].append(self.valve_out_B203)
                self.log['valve_out_B204'].append(self.valve_out_B204)
                self.log['pump_alt'].append(self.pump_alt*self.pump_power_alt)
                self.log['pump_main'].append(self.pump_main*self.pump_power)
                self.log['condition'].append(self.condition)
                self.check_level()
                self.get_state('state_emptying_tank_B203')
                self.get_volumetric_flow_rate()
                if self.condition == 'sensor_fault':
                    self.log['sensor_continuous_tank'].append(water_pumped*(random.uniform(0.8,0.9)))
                else:
                    self.log['sensor_continuous_tank'].append(water_pumped)
                if self.level_B203 <= self.min_height_B203:
                    print('Tank B203 is empty.')
                    # self.valve_out_B203 = 0
                    # self.pump_power_alt = 0
                    self.log['valve_out_B202'][-1] = self.valve_out_B203
                    return self.log
            else:
                print('Error')
                valves={'valve_in_B204':self.valve_in_B204,'valve_out_B203':self.valve_out_B203,'pump_main':self.pump_main,'pump_alt':self.pump_alt}
                error_valves = [i for i in valves if valves[i]!=True]
                raise ValueError(f"{error_valves} is closed, cannot empty the tankB203.")
                
    def state_emptying_tank_B204(self):
        '''
            This method updates the state of tank B203 and B204 based on the current state and the valve settings.
        '''
        time_limit = int(min(self.max_time-self.time, self.filling_tank_time))
        print('time_limit:', time_limit)
        noise = +random.uniform(-0.00002, 0.00002)
        while self.level_B204>self.min_height_B204 and time_limit>0:
            self.time +=1/10
            if self.valve_out_B204:
                self.level_B204 -= 0.001 + random.uniform(-0.0002, 0.0002)
                self.log['time'].append(self.get_time())
                self.log['level_B201'].append(self.level_B201)
                self.log['level_B202'].append(self.level_B202)
                self.log['level_B203'].append(self.level_B203)
                self.log['level_B204'].append(self.level_B204)
                self.log['valve_in_B201'].append(self.valve_in_B201)
                self.log['valve_in_B202'].append(self.valve_in_B202)
                self.log['valve_in_B203'].append(self.valve_in_B203)
                self.log['valve_in_B204'].append(self.valve_in_B204)
                self.log['valve_out_B201'].append(self.valve_out_B201)
                self.log['valve_out_B202'].append(self.valve_out_B202)
                self.log['valve_out_B203'].append(self.valve_out_B203)
                self.log['valve_out_B204'].append(self.valve_out_B204)
                self.log['pump_alt'].append(self.pump_alt*self.pump_power_alt)
                self.log['pump_main'].append(self.pump_main*self.pump_power)
                self.log['condition'].append(self.condition)
                self.check_level()
                self.get_state('state_emptying_tank_B204')
                self.get_volumetric_flow_rate()
                self.log['sensor_continuous_tank'].append(0)
                if self.level_B204 <= self.min_height_B204:
                    print('Tank B204 is empty.')
                    self.valve_out_B204 = 0
                    self.valve_in_B201 = 1
                    self.log['valve_out_B204'][-1] = self.valve_out_B204
                    return self.log
                
            else:
                print('Error')
                valves={'valve_out_B204':self.valve_out_B204}
                error_valves = [i for i in valves if valves[i]!=True]
                raise ValueError(f"{error_valves} is closed, cannot empty the tankB204.")
                
    
    
            
         
    
            
