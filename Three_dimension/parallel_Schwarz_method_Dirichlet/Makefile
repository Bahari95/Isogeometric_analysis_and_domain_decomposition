VAR='execute vtk.py and clean the .npy'
VAR1='python execution'
VAR2='after vtk execution clean .npy'

execute:
	@python vtk.py
	
middle:execute
	@echo ${VAR} 
cleaning : middle
	@rm *.npy

