# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 11:55:19 2016

@author: emile.roux@univ-smb.fr
"""
from mpl_toolkits.axes_grid.axislines import SubplotZero
from matplotlib.transforms import BlendedGenericTransform
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

##Construction of the Morh circle for an in-plan stress state.
##The morh circle is lived display using animation feature of matplotlib.
##
##Input : the stress tensor
##Output : anmiation

# ########################
# INPUT : Stress tensor ! must be a plan stress state !
# ########################
Sigma=np.array([[400,100,0],[100,-200, 0 ],[0,0, 0 ]],float)
print("Sigma=", Sigma)

# ########################
# Set up the figure, the axis, and the plot element we want to animate
# ########################
fig = plt.figure(1)
Sig_max=500.0
# Espace x,y
ax1 = SubplotZero(fig, 121)
fig.add_subplot(ax1)
#
for direction in ["xzero", "yzero"]:
    ax1.axis[direction].set_axisline_style("-|>")
    ax1.axis[direction].set_visible(True)
#
for direction in ["left", "right", "bottom", "top"]:
    ax1.axis[direction].set_visible(False)
    
ax1.set_aspect('equal')

ax1.set_xlim(-Sig_max, Sig_max)
ax1.set_ylim(-Sig_max, Sig_max)
ax1.text(0., 1.05, 'y',size=20, transform=BlendedGenericTransform(ax1.transData, ax1.transAxes))
ax1.text(1.05, -0.15, 'x',size=20, transform=BlendedGenericTransform(ax1.transAxes, ax1.transData))

vec_phi_xy  = ax1.quiver(0, 0, 0, 0,width=10,scale=1,units='x',label=r'$\phi_n$',color='b')
vec_Snn_xy  = ax1.quiver(0, 0, 0, 0,width=6,scale=1,units='x',label=r'$\sigma_{nn}$',color='c')
vec_Snt_xy  = ax1.quiver(0, 0, 0, 0,width=6,scale=1,units='x',label=r'$\sigma_{nt}$',color=(1.0,0.25,0.75))
line_proj,  = ax1.plot([],[],'--')

facette,= ax1.plot([],[],'--g',lw=2)
vec_n  = ax1.quiver(0, 0, 0, 0,width=4,scale=4/Sig_max,units='x',label=r'$n$',color='g') 
ax1.legend()


# Espace Snn,Snt
ax2 = SubplotZero(fig, 122)
fig.add_subplot(ax2)
#
for direction in ["xzero", "yzero"]:
    ax2.axis[direction].set_axisline_style("-|>")
    ax2.axis[direction].set_visible(True)
#
for direction in ["left", "right", "bottom", "top"]:
    ax2.axis[direction].set_visible(False)
    
ax2.set_aspect('equal')

ax2.set_xlim(-Sig_max, Sig_max)
ax2.set_ylim(-Sig_max, Sig_max)
ax2.text(0., 1.05, '$\sigma_{nt}$',size=20, transform=BlendedGenericTransform(ax2.transData, ax2.transAxes))
ax2.text(1.05, -0.15, '$\sigma_{nn}$',size=20, transform=BlendedGenericTransform(ax2.transAxes, ax2.transData))

mohr_circle, = ax2.plot([], [], '-+r',label='Mohr Circle')
vec_phi  = ax2.quiver(0, 0, 0, 0,width=10,scale=1,units='x',label=r'$\phi_n$',color='b')
ax2.legend()



# ########################

def Snn_Snt_decomposition (Sigma, n) :
    # Check that niomr of n is one
    n=n/np.linalg.norm(n)
    
    # Flux stress vectir
    phi=np.dot(Sigma,n)
    
    # Normal stress component
    Snn=np.dot(phi,n)

    
    # Shear stress component
    t=np.array([-n[1], n[0], n[2]])
    Snt=np.dot(phi,t)
#    Snt=phi - Snn*n
#    Snt=np.linalg.norm(Snt)  
    
    return Snn,Snt,phi


# ########################
# initialization function: plot the background of each frame
def init():
    mohr_circle.set_data([], [])
    return mohr_circle,

# #######################
# animation function.  This is called sequentially
def animate(i):
    # Stuied normal
    a= 2*i * np.pi/180
    n=np.array([np.cos(a), np.sin(a), 0])
    Snn,Snt,phi = Snn_Snt_decomposition (Sigma, n)
    
    # espace x, y
    vec_phi_xy.set_UVC(phi[0],phi[1])
    vec_n.set_UVC(n[0],n[1])
    fac_x=np.array([n[1],-n[1]])
    fac_y=np.array([-n[0],n[0]])
    facette.set_data(50*fac_x,50*fac_y)
    
    vec_Snn_xy.set_UVC(n[0]*Snn,n[1]*Snn)
    vec_Snt_xy.set_UVC(-n[1]*Snt,n[0]*Snt)
    line_proj.set_data([-n[1]*Snt,phi[0],n[0]*Snn],[n[0]*Snt,phi[1],n[1]*Snn])
    
    

    
    # espace Snn, Snt
    x,y=mohr_circle.get_data()
    mohr_circle.set_data(np.append(x,Snn), np.append(y,Snt))
    vec_phi.set_UVC(Snn,Snt)

    return mohr_circle,vec_phi_xy, vec_n,fac_x,fac_y,facette,vec_Snn_xy,vec_Snt_xy,line_proj,vec_phi

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, 
                               frames=90, interval=200, blit=False)


plt.show()

