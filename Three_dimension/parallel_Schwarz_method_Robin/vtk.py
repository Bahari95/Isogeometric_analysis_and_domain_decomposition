from evtk.hl import gridToVTK 
import numpy as np 
import random as rnd 

# Coordinates
x = np.load("mesh_x.npy")
y = np.load("mesh_y.npy")
z = np.load("mesh_z.npy")
# We add Jacobian function to make the grid more interesting
Solut_uh  = np.load("uh.npy")
Solut_uhx = np.load("uhx.npy")
Solut_uhy = np.load("uhy.npy")
Solut_uhz = np.load("uhz.npy")
Solut_u   = np.load("Sol.npy")
# Dimensions 
nx, ny, nz = x.shape[0], y.shape[0], z.shape[0]
ncells = nx * ny * nz
npoints = (nx + 1) * (ny + 1) * (nz + 1)

# Variables <br>
gridToVTK("./domain", x, y, z, pointData = {"Solut_uh" : Solut_uh, "Solut_uhx" : Solut_uhx,"Solut_uhy" : Solut_uhy, "Solut_uhz" : Solut_uhz, "Solut_u" : Solut_u})

# Coordinates_1
x = np.load("mesh_x_1.npy")
y = np.load("mesh_y_1.npy")
z = np.load("mesh_z_1.npy")
# We add Jacobian function to make the grid more interesting
Solut_uh  = np.load("uh_1.npy")
Solut_uhx = np.load("uhx_1.npy")
Solut_uhy = np.load("uhy_1.npy")
Solut_uhz = np.load("uhz_1.npy")
Solut_u   = np.load("Sol_1.npy")
# Dimensions 
nx, ny, nz = x.shape[0], y.shape[0], z.shape[0]
ncells = nx * ny * nz
npoints = (nx + 1) * (ny + 1) * (nz + 1)

# Variables <br>
gridToVTK("./domain_1", x, y, z, pointData = {"Solut_uh_1" : Solut_uh, "Solut_uhx_1" : Solut_uhx,"Solut_uhy_1" : Solut_uhy, "Solut_uhz_1" : Solut_uhz, "Solut_u_1" : Solut_u})
