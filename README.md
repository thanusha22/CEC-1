
## Features
- Hybrid nature-inspired metaheuristic optimizer alond with other algorithms were implemented.
- The implimentation uses the fast array manipulation using `NumPy`.
- Matrix support using `SciPy`'s package.
- More optimizers is comming soon.

## Installation
- Python 3.xx is required.

Run

    pip3 install -r requirements.txt

(possibly with `sudo`)

That command above will install  `sklearn`, `NumPy`, and `SciPy` for
you.



## Get the source

Clone the Git repository from GitHub

    git clone https://github.com/ashugowda/Evolopy.git


## Quick User Guide


Select optimizers from the list of available ones: "HYBRID","SSA","PSO","BAT","FFA","GWO","WOA","MVO","MFO","HHO","SCA","JAYA". For example:
```
optimizer=["SSA","PSO"]  
```

After that, Select benchmark function from the list of available ones: "F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F14","F15","F16","F17","F18","F19". For example:
```
objectivefunc=["F3","F4"]  
```

Select number of repetitions for each experiment. To obtain meaningful statistical results, usually 30 independent runs are executed for each algorithm.  For example:
```
NumOfRuns=10  
```
Select general parameters for all optimizers (population size, number of iterations). For example:
```
params = {'PopulationSize' : 30, 'Iterations' : 50}
```
Choose whether to Export the results in different formats. For example:
```
export_flags = {'Export_avg':True, 'Export_details':True, 'Export_convergence':True, 'Export_boxplot':True}
```

Run the code!






