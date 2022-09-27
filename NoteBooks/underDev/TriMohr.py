# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 10:07:27 2017

@author: rouxemi
"""
from mpl_toolkits.axes_grid.axislines import SubplotZero
from matplotlib.transforms import BlendedGenericTransform
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import matplotlib.patches as patches

##Construction of the Tri Morh circle for a given stress state.
##The morh circle is lived display using animation feature of matplotlib.
##
##Input : the stress tensor
##Output : anmiation

# ########################
# INPUT : Stress tensor
# Norme componants
S11=2
S22=4
S33=0
#Shear Componants
S12=0
S13=0
S23=0
# ########################
Sigma=np.array([[S11,S12,S13],[S12,S22, S23 ],[S13,S23, S33 ]],float)
print("Sigma=", Sigma)
VP=np.linalg.eigvals(Sigma)
Sig_max= np.max((VP)) + 5
Sig_min= np.min((VP)) - 5 


# ########################
# Set up the figure, the axis, and the plot element we want to animate
# ########################
fig = plt.figure(1)

# Espace x,y
# Espace Snn,Snt
ax2 = SubplotZero(fig, 111)
fig.add_subplot(ax2)
#
for direction in ["xzero", "yzero"]:
    ax2.axis[direction].set_axisline_style("-|>")
    ax2.axis[direction].set_visible(True)
#
for direction in ["left", "right", "bottom", "top"]:
    ax2.axis[direction].set_visible(False)
    
ax2.set_aspect('equal')

ax2.set_xlim(Sig_min, Sig_max)
ax2.set_ylim(-(Sig_max-Sig_min)/2, (Sig_max-Sig_min)/2)
ax2.text(0., 1.05, '$\sigma_{nt}$',size=20, transform=BlendedGenericTransform(ax2.transData, ax2.transAxes))
ax2.text(1.05, -0.15, '$\sigma_{nn}$',size=20, transform=BlendedGenericTransform(ax2.transAxes, ax2.transData))
ax2.grid()
mohr_circle, = ax2.plot([], [], '.r',label='Etat de contrainte', markersize=2)

ax2.legend()


# ########################

def Snn_Snt_decomposition (Sigma, n) :
    # Check that norm of n is one
    n=n/np.linalg.norm(n)
    
    # Flux stress vectir
    phi=np.dot(Sigma,n)
    
    # Normal stress component
    Snn=np.dot(phi,n)
    
    # Shear stress component
    Snt=phi - Snn*n
    Snt=np.linalg.norm(Snt)  
    
    return Snn,Snt,phi


# ########################
    
  
# #######################
# animation function.  This is called sequentially
def animate(i):
  for j in range(10):
    n=np.random.rand(3)-.5
    Snn,Snt,phi = Snn_Snt_decomposition (Sigma, n)
    x,y=mohr_circle.get_data()
    mohr_circle.set_data(np.append(x,[Snn, Snn]), np.append(y,[Snt, -Snt]))
    # add Eigen values on graph:
    if i==50:
      ax2.plot(VP,[0, 0, 0],'og', label='Contrainte Princ')
      #ax2.legend()
  return mohr_circle

# call the animator.  blit=True means only re-draw the parts that have changed.

anim = animation.FuncAnimation(fig, animate, 
                               frames=100, interval=200., blit=False,repeat=False)

print("Vp=",VP)
plt.show()  


  

  
  
  
  
  
  
  
  
  
  
  
  
  