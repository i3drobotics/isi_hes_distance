import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd

def func(x, a, b):
    return a * x + b

#importing data from the excel sheet
df = pd.read_excel ('Titaniumoxidedata.xls')
Laser_alt = df['Laser']
Deimos1_alt = df['Deimos1']
Deimos2_alt = df['Deimos2']
Deimos3_alt = df['Deimos3']
Corrected_alt = df['Average']
Laser = []
Deimos1 = []
Deimos2 = []
Deimos3 = []
Corrected = []

for L in range(len(Laser_alt)):
    Laser.append(Laser_alt[L])

for D1 in range(len(Deimos1_alt)):
    Deimos1.append(Deimos1_alt[D1])
    
for D2 in range(len(Deimos2_alt)):
    Deimos2.append(Deimos2_alt[D2])
    
for D3 in range(len(Deimos3_alt)):
    Deimos3.append(Deimos3_alt[D3])

for Co in range(len(Corrected_alt)):
    Corrected.append(Corrected_alt[Co])

popt,pcov=curve_fit(func,Laser,Deimos1)

error = np.sqrt(np.diag(pcov))
print('Deimos repeat 1')
print('a =',popt[0],'+/-',error[0])
print('b =',popt[1],'+/-',error[1])
print()

popt,pcov=curve_fit(func,Laser,Deimos2)

error = np.sqrt(np.diag(pcov))
print('Deimos repeat 2')
print('a =',popt[0],'+/-',error[0])
print('b =',popt[1],'+/-',error[1])
print()

popt,pcov=curve_fit(func,Laser,Deimos3)

error = np.sqrt(np.diag(pcov))
print('Deimos repeat 3')
print('a =',popt[0],'+/-',error[0])
print('b =',popt[1],'+/-',error[1])
print()

popt,pcov=curve_fit(func,Laser,Corrected)

error = np.sqrt(np.diag(pcov))
print('Corrected')
print('a =',popt[0],'+/-',error[0])
print('b =',popt[1],'+/-',error[1])
print()





