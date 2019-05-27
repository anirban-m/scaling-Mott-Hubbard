import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from Electrons import Electrons
fig0 , ax0 = plt.subplots(ncols=1)
fig, ax = plt.subplots(ncols=1)
fig1, ax1 = plt.subplots(ncols=1)
import seaborn as sns
sns.set_style('white')
sns.set_style('ticks')
Lambda0 = 0.257
def process(N,RGsteps):
	
	intvl = np.pi/N
	dk = 1
	kx, ky = intvl*np.mgrid[slice(0, N+dk, dk),slice(0, N+dk, dk)]
	
        #make the lattice
	k = (kx,ky)
	#Call the class electrons(e)
	electrons = Electrons()
	Ek = np.float64(-2*(np.cos(kx)+np.cos(ky)))
	#bandwidth	
	W = 8 
	#Show the Fermi surface(F) and the volume it bounds
	cs = ax0.contour(k[0][:-1,:-1]+(intvl/2.)*dk,k[1][:-1,:-1]+(intvl/2.)*dk,Ek[:-1,:-1],levels=[0],colors='red') 
	#Call axes artist to get the coordinates identify the Fermi surface(F) wavefront(wvf)
	F = cs.collections[0].get_paths()[0]
	# (F) curve
	kx,ky,k,Ek=np.nan,np.nan,np.nan,np.nan
	cs=np.nan
	coordsF = F.vertices
	#transpose (F) curve
	coordsF_T = np.flip(coordsF,axis = 1)
	#Vel. vec. vF at (F)
	vF = electrons.vk(coordsF)
	vF_T = electrons.vk(coordsF_T)
	vFmag = vF[:,0]*vF[:,0]+vF[:,1]*vF[:,1]
	shatF  = vF/np.transpose([vFmag]) 
	#Fermi surface angular coordinates, choosing the X=Y symmetry and plotting one part
	thetaF0 =  np.float64(np.arctan(vF[:,1]/vF[:,0])) 
	thetaF = thetaF0[thetaF0>=np.pi/4]
	#making a list of offsets around the Fermi surface from 0 to $\Lambda_{0}$
	lambdaF = Lambda0*np.linspace(0.99,1,RGsteps+1)
	#making a mesh out of the wave fronts in curvilinear coords thetaF,lambdaF
	ThetaF,LambdaF = np.meshgrid(thetaF,lambdaF)
	coordsFx_irr = coordsF[:,0][thetaF0>=np.pi/4]
	coordsFy_irr = coordsF[:,1][thetaF0>=np.pi/4]
	Fwvfx_out = coordsFx_irr+LambdaF*np.cos(ThetaF)
	Fwvfy_out = coordsFy_irr+LambdaF*np.sin(ThetaF)
	Fwvfx_in  = coordsFy_irr-LambdaF*np.cos(ThetaF) 
	Fwvfy_in  = coordsFx_irr-LambdaF*np.sin(ThetaF) 
		
	Fwvfy_out[Fwvfy_out>np.pi] = np.pi
	Fwvfy_out[Fwvfy_out<0] = 0
	Fwvfx_out[Fwvfx_out>np.pi] = np.pi
	Fwvfx_out[Fwvfx_out<0] = 0
	Fwvfy_in[Fwvfy_in>np.pi] = np.pi
	Fwvfy_in[Fwvfy_in<0] = 0
	Fwvfx_in[Fwvfx_in>np.pi] = np.pi
	Fwvfx_in[Fwvfx_in<0] = 0
	Fwvf_out = (Fwvfx_out,Fwvfy_out)
	Fwvf_in = (Fwvfx_in,Fwvfy_in)

#Getting the electronic dispersion for outgoing (wvf) around (F)-pair 1
	EkFwvf_out = electrons.Ek(Fwvf_out)
#Getting the electronic disperfsion for incoming (wvf) around (F)-pair 1	
	EkFwvf_in = electrons.Ek(Fwvf_in)
	#U0 = 8t
	U0 = 8./(N)
	omegaSat =2*(EkFwvf_out[:,len(thetaF)-1]-EkFwvf_in[:,len(thetaF)-1]).max()
	invGgapped = omegaSat - 2*(EkFwvf_out[:,len(thetaF)-1]  -EkFwvf_in[:,len(thetaF)-1])
	invGint = invGgapped - U0
	lambdaEm = lambdaF[np.argmin(invGint>0)]
	renV = omegaSat-2*EkFwvf_out[:,len(thetaF)-1][lambdaF==lambdaEm]+2*EkFwvf_in[:,len(thetaF)-1][lambdaF==lambdaEm]
	energyPerParticle = -renV*lambdaEm*lambdaEm*N
	print("System size, Energy per particle",N,energyPerParticle)
	return energyPerParticle
N_latt = np.array([256,512,1024,2048,4096,8192,16384,32768])
RGstepL = np.array([1000,1000,1000,1000,1000,1000,10000,10000])
energyGSperParticle = 0.0*N_latt
energyGSperParticle = np.asarray([process(N,RGstepL[N_latt==N]) for N in N_latt])
#plot for ground-state-energy per particle with inverse square root lattice size  	
ax.plot(1/(N_latt+0.0),energyGSperParticle, marker='o',color='red')
ax.set_xlabel(r"$1/\sqrt{vol}$")
ax.set_ylabel(r"$E_{g}$")	
plt.show()

