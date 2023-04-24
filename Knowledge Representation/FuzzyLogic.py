import numpy as np
from skfuzzy import control as ctrl
import skfuzzy as fuzz

def fuzzy(weather_info):

    # Define input variables
    temp =   ctrl.Antecedent(np.arange(-50, 101, 1), 'temperature')
    visib =  ctrl.Antecedent(np.arange(0, 11, 1), 'visibility')
    precip = ctrl.Antecedent(np.arange(0, 11, 1), 'precipitation')
    wind =   ctrl.Antecedent(np.arange(0, 101, 1), 'windspeed')

    # Define output variable
    delay_multiplier = ctrl.Consequent(np.arange(0, 3.01, 0.01), 'delay_multiplier') 

    # Define membership functions for input variables
    temp['cold'] = fuzz.trimf(temp.universe, [-20, 0, 20])
    temp['moderate'] = fuzz.trimf(temp.universe, [10, 30, 50])
    temp['hot'] = fuzz.trimf(temp.universe, [40, 80, 100])

    visib['poor'] = fuzz.trimf(visib.universe, [0, 0, 5])
    visib['fair'] = fuzz.trimf(visib.universe, [2, 5, 8])
    visib['good'] = fuzz.trimf(visib.universe, [6, 10, 10])

    precip['none'] = fuzz.trimf(precip.universe, [0, 0, 1])
    precip['light'] = fuzz.trimf(precip.universe, [0, 2.5, 5])
    precip['heavy'] = fuzz.trimf(precip.universe, [4, 10, 10])

    wind['calm'] = fuzz.trimf(wind.universe, [0, 0, 20])
    wind['moderate'] = fuzz.trimf(wind.universe, [10, 40, 70])
    wind['strong'] = fuzz.trimf(wind.universe, [60, 100, 100])

    # Define membership functions for output variable
    
    delay_multiplier['low'] = fuzz.trimf(delay_multiplier.universe, [0, 0, 1])
    delay_multiplier['medium'] = fuzz.trimf(delay_multiplier.universe, [0.5, 1.125, 1.75])
    delay_multiplier['high'] = fuzz.trimf(delay_multiplier.universe, [1.125, 1.5, 1.5])


    # Define fuzzy rules
    rule1 = ctrl.Rule(temp['cold'] & visib['poor'], delay_multiplier['high'])
    rule2 = ctrl.Rule(temp['cold'] & visib['fair'], delay_multiplier['medium'])
    rule3 = ctrl.Rule(temp['cold'] & visib['good'], delay_multiplier['low'])
    rule4 = ctrl.Rule(temp['moderate'] & visib['poor'], delay_multiplier['high'])
    rule5 = ctrl.Rule(temp['moderate'] & visib['fair'], delay_multiplier['medium'])
    rule6 = ctrl.Rule(temp['moderate'] & visib['good'], delay_multiplier['low'])
    rule7 = ctrl.Rule(temp['hot'] & visib['poor'], delay_multiplier['high'])
    rule8 = ctrl.Rule(temp['hot'] & visib['fair'], delay_multiplier['medium'])
    rule9 = ctrl.Rule(temp['hot'] & visib['good'], delay_multiplier['low'])
    rule10 = ctrl.Rule(precip['none'] & wind['calm'], delay_multiplier['low'])
    rule11 = ctrl.Rule(precip['none'] & wind['moderate'], delay_multiplier['low'])
    rule12 = ctrl.Rule(precip['none'] & wind['strong'], delay_multiplier['low'])
    rule13 = ctrl.Rule(precip['light'] & wind['calm'], delay_multiplier['medium'])
    rule14 = ctrl.Rule(precip['light'] & wind['moderate'], delay_multiplier['medium'])
    rule15 = ctrl.Rule(precip['light'] & wind['strong'], delay_multiplier['medium'])
    rule16 = ctrl.Rule(precip['heavy'] & wind['calm'], delay_multiplier['high'])
    rule17 = ctrl.Rule(precip['heavy'] & wind['moderate'], delay_multiplier['medium'])
    rule18 = ctrl.Rule(precip['heavy'] & wind['strong'], delay_multiplier['medium'])


    #Define Fuzzy system
    delay_multiplier_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18])
    delay_multiplier_simulation = ctrl.ControlSystemSimulation(delay_multiplier_ctrl)

    #Set Input Values
    delay_multiplier_simulation.input['temperature'] = weather_info['temp']
    delay_multiplier_simulation.input['visibility'] = weather_info['vis']
    delay_multiplier_simulation.input['precipitation'] = weather_info['precip']
    delay_multiplier_simulation.input['windspeed'] = weather_info['wind']

    #Compute Output
    delay_multiplier_simulation.compute()
    return delay_multiplier_simulation.output['delay_multiplier']