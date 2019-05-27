import numpy as np
import matplotlib
from time import time
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from Electrons import Electrons
fig, ax0 = plt.subplots(ncols=1)
fig, ax = plt.subplots(ncols=1)
fig1, ax1 = plt.subplots(ncols=1)
fig2, ax2 = plt.subplots(ncols=1)
import seaborn as sns
sns.set_style('white')
sns.set_style('ticks')
Lambda0 = 0.257 
ti=time()
N = 1024
intvl = np.pi/N
dk = 1
kx, ky = intvl*np.mgrid[slice(0, N+dk, dk),slice(0, N+dk, dk)]
k = (kx,ky)
electrons = Electrons()
Ek = np.float64(-2*(np.cos(kx)+np.cos(ky)))
W = 8 
cs = ax0.contour(k[0][:-1,:-1]+(intvl/2.)*dk,k[1][:-1,:-1]+(intvl/2.)*dk,Ek[:-1,:-1],levels=[0],colors='red') 		
F = cs.collections[0].get_paths()[0]
ax.clear()
kx,ky,k,Ek=np.nan,np.nan,np.nan,np.nan
cs=np.nan
coordsF = F.vertices
coordsF_T = np.flip(coordsF,axis = 1)
vF = electrons.vk(coordsF)
vF_T = electrons.vk(coordsF_T)
vFmag = vF[:,0]*vF[:,0]+vF[:,1]*vF[:,1]
shatF  = vF/np.transpose([vFmag]) 
thetaF0 =  np.float64(np.arctan(vF[:,1]/vF[:,0]))
thetaF = thetaF0[thetaF0>=np.pi/4]
RGsteps = 1000
lambdaF = Lambda0*np.linspace(0,1,RGsteps+1)
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
EkFwvf_out = electrons.Ek(Fwvf_out)	
EkFwvf_in = electrons.Ek(Fwvf_in)
U0 = 8./(N)
omegaM = W/2.
										#-------------------Doped Mott liquid----------------------- 
muEff = np.sort(np.concatenate((np.linspace(0.000,-3.95,100),np.array([-1.997]))))[::-1] 
pstar = 0.0*muEff
#Function determines the spin charge hybridization coefficient given argument mu
def spinChargeInterplay(mu):
	p = np.linspace(0,1,2000)
	Echarge = 2*(EkFwvf_out[:,len(thetaF)-1][lambdaF==Lambda0]+EkFwvf_in[:,len(thetaF)-1][lambdaF==Lambda0])-mu
	Espin = 2*(EkFwvf_out[:,len(thetaF)-1][lambdaF==Lambda0]-EkFwvf_in[:,len(thetaF)-1][lambdaF==Lambda0])	
	invG = omegaM -Echarge*p-Espin*(1-p)
	return p[invG==invG[(invG-U0)>0].min()]
pstar = np.asarray([spinChargeInterplay(mu) for mu in muEff])
#Function determines the saturation quantum fluctn. at which Fermi surface vanishes
def omegaSaturation(mu,p):
	Echarge = 2*(EkFwvf_out[:,len(thetaF)-1]+EkFwvf_in[:,len(thetaF)-1])-mu
	Espin = 2*(EkFwvf_out[:,len(thetaF)-1]-EkFwvf_in[:,len(thetaF)-1])
	EspHybrid = p*Echarge+(1-p)*Espin
	maxEspHybrid = np.max(EspHybrid)
	deltaEspHybrid = maxEspHybrid-EspHybrid
	if (deltaEspHybrid[deltaEspHybrid>U0].size==0 and U0+np.min(EspHybrid)<omegaM and omegaM+mu>0):
		return U0+np.min(EspHybrid)
	elif(deltaEspHybrid[deltaEspHybrid>U0].size!=0 and omegaM+mu>0):		
		return maxEspHybrid
	else:		
		return omegaM 
omegaStar = np.asarray([omegaSaturation(mu,pstar[muEff==mu][0]) for mu in muEff])
#Function determines the RG fixed point distances from the Fermi surface along the nodal direction
def RGiteratorDML(omega,mu,p):
	if omega+mu>0:
		Echarge = 2*(EkFwvf_out[:,len(thetaF)-1]+EkFwvf_in[:,len(thetaF)-1])-mu
		Espin = 2*(EkFwvf_out[:,len(thetaF)-1]-EkFwvf_in[:,len(thetaF)-1])
		invG = (omega -Echarge*p-Espin*(1-p)-U0)[0]
		lambdaEm = np.max(lambdaF[np.where(np.diff(np.sign(invG)))[0]+1])
		return lambdaEm
	else:
		return np.nan
lambdaEmArr = np.asarray([RGiteratorDML(omegaStar[muEff==mu],mu,pstar[muEff==mu]) for mu in muEff])
#Array stores the total number of states in emergent window
NstarArrInt = np.asarray([(lambdaEmArr[muEff==mu]*N).astype(int) if omegaStar[muEff==mu]+mu>0 else np.nan for mu in muEff])		
eGS = 0.0*lambdaEmArr.copy()
fill = 0.0*eGS
def detEnergyAndFilling(Kc, Ks, mu, Nstar,N):
	Karr = np.linspace(0,Nstar,Nstar+1).astype(int)
	nArr = np.asarray([np.linspace(0,2*K,2*K+1).astype(int) for K in Karr])
	EArr =0.0*nArr
	for j in range(len(nArr)):
		nK = nArr[j]
		K = int((np.shape(nK)[0]-1)/2)   
		for k in range(len(nK)):
			#0<K<Nstar, 0=<l=<2Nstar-2K, -l=<p=<l, 0=<n=<2K, -n=<m=<n 
			EArr[j][k] = -Ks*(Nstar-K)*(Nstar-K) +(Kc/2.)*(nK[k]*(nK[k]+1)-2*K*(K+1)) + mu*nK[k]
	EminK = np.zeros(np.shape(EArr))
	fillK = np.zeros(np.shape(EArr))
	for l in range(len(nArr)):
		EminK[l] = np.min(EArr[l])
		fillK[l] = nArr[l][np.argmin(EArr[l])]
	return np.min(EminK)/N, fillK[np.argmin(EminK)]/(2*N)
for i in range(len(muEff)):
	Nstar = NstarArrInt[i]
	mu = muEff[i]
	eGS[i] = detEnergyAndFilling(U0,U0, mu, Nstar,N)[0]
	fill[i] = detEnergyAndFilling(U0,U0, mu, Nstar,N)[1]
	print("chemical potential, doping fraction, Eg:-",mu,fill[i],eGS[i])



#parameters for n=0.1245 doping
pstar1 = pstar[muEff==-1.997]
mu1 = -1.997
print("hello",pstar1)

print ('_____________________D-O-N-E--Q-C-P_______________________')
#-----------------------beyond MOTT QCP------------------------- 
muEff1 = np.concatenate([np.linspace(-4.00,-5.00,250),np.linspace(-5.01,-8,70)])
p = np.linspace(0,1,2000)
pstar = 0.0*muEff1
Theta,P_Theta = np.meshgrid(ThetaF[0],p)
for mu in muEff1:
	invG = omegaM +mu*p
	pstar[muEff1==mu] = p[invG==invG[(invG-U0)>0].min()]
omega = W/2.
def bsANrgIteratorBeyondQCP(omega, mu):
	p = pstar[muEff1 == mu]
	invGapped = omega+mu*p-0.5*((1-p)*(EkFwvf_out[len(lambdaF)-1,:] -EkFwvf_in[len(lambdaF)-1,:])+p*(EkFwvf_out[len(lambdaF)-1,:]  +EkFwvf_in[len(lambdaF)-1,:]))
	fracGapped = (len(invGapped[invGapped>0]))/len(thetaF)
	invGgappedAN =omega+mu*p-0.5*((1-p)*(EkFwvf_out[:,0] -EkFwvf_in[:,0])+p*(EkFwvf_out[:,0]  +EkFwvf_in[:,0])) 
	KcAN, KsAN = 0.0*invGgappedAN.copy(), 0.0*invGgappedAN.copy()
	KeffAN = p*KcAN -(1-p)*KsAN
	KsAN[len(lambdaF)-1],KcAN[len(lambdaF)-1] = U0, U0
	if invGgappedAN[len(lambdaF)-1]>0:
		for i in range(1,len(lambdaF))[::-1]:
			KsAN[i-1] = KsAN[i] -(1-p)*KsAN[i]*KsAN[i]/(invGgappedAN[i]-KeffAN[i]/4.)
			KcAN[i-1] = (KsAN[i-1]*U0*p)/(KsAN[i-1]-U0*(1-p))     
			KeffAN[i-1]  = p*KcAN[i-1]-(1-p)*KsAN[i-1]
			if invGgappedAN[i-1]-KeffAN[i-1]/4.<0 or (KsAN[i-1]-U0*(1-p))<0:  
				lambdaMuAN = lambdaF[i-1]	
				KsMuAN = KsAN[i]
				KcMuAN = KcAN[i]
				return lambdaMuAN,KsMuAN,KcMuAN,fracGapped
			else:
				return Lambda0,U0,U0,fracGapped
	else:
		return 0.,0.,0.,0.	
def fsNrgIteratorBeyondQCP(omega, mu):
	p = pstar[muEff1 == mu]
	invGgaplessN =-omega-mu*p+0.5*((1-p)*(EkFwvf_out[:,0] -EkFwvf_in[:,0])+p*(EkFwvf_out[:,0]  +EkFwvf_in[:,0])) 
	KcN, KsN = 0.0*invGgaplessN.copy(), 0.0*invGgaplessN.copy()
	KeffN = p*KcN -(1-p)*KsN
	KsN[len(lambdaF)-1],KcN[len(lambdaF)-1] = U0, U0
	for i in range(1,len(lambdaF))[::-1]:
		KsN[i-1] = KsN[i] -(1-p)*KsN[i]*KsN[i]/(invGgaplessN[i]-KeffN[i]/4.)
		KcN[i-1] = (KsN[i-1]*U0*p)/(KsN[i-1]-U0*(1-p))	
		KeffN[i-1] = p*KcN[i-1]-(1-p)*KsN[i-1]
		if invGgaplessN[i-1]-KeffN[i-1]/4.<0 or KsN[i-1]-U0*(1-p)<0:
			lambdaMuN = lambdaF[i-1]				
			KsN[i-1] = 0
			return lambdaMuN
		else:
			return 0.0
def TSrgIteratorBeyondQCP(omega, mu):
	L = 0.0*lambdaF.copy()
	L[len(L)-1]=U0
	for i in range(1,len(lambdaF))[::-1]:
		L[i-1] = L[i] + len(thetaF)*(len(thetaF)+1)*L[i]*L[i]/(omega-W-mu-L[i]/4.)	
		if (omega-W-mu-L[i-1]/4.<0):
			return lambdaF[i-1]
		else:
			return 0.0
eGS1 = 0.0*muEff1
fh = 0.0*muEff1 
for mu in muEff1:
	lambdaMuAN, KsMuAN, KcMuAN, fracGapped = bsANrgIteratorBeyondQCP(omega, mu)
	lambdaMuN = fsNrgIteratorBeyondQCP(omega, mu)
	lambdaMuT = TSrgIteratorBeyondQCP(omega, mu)	
	Nstar = int(lambdaMuAN*N)
	Eg, fill0 = detEnergyAndFilling(KsMuAN, KcMuAN, mu, Nstar,N) 	
	eGS1[muEff1==mu] = Eg*fracGapped
	fill1 = 0.5*(lambdaMuN/Lambda0)
	fill2 =  0.5*(lambdaMuT/Lambda0)
	fh[muEff1==mu] = fill0*fracGapped+(1-fracGapped)*(fill1+(1-np.sign(fill1))*fill2+(1-np.sign(fill1))*(1-np.sign(fill2)))
	print("chemical potential, doping fraction, Eg-",mu,fh[muEff1==mu],eGS1)
EgComplete = np.concatenate([eGS,eGS1])						
fhComplete = np.concatenate([fill,fh])
muComplete = np.concatenate([muEff,muEff1])
print(time()-ti)
ax.plot(-muComplete,fhComplete,linestyle='',marker='o')
ax.set_ylabel(r"$f_{h}$")
ax.set_xlabel(r"$-\Delta\mu_{eff}$(in units of t)")
ax1.plot(fhComplete,EgComplete,linestyle='',marker='o')
ax1.set_xlabel(r"$f_{h}$(doping fraction)")
ax1.set_ylabel(r"$E_{g}$(in units of t)") 	
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
	lambdaF = Lambda0*np.linspace(0,1,RGsteps+1)
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

	#Getting the electronic dispersion for states outside Fermi surface
	EkFwvf_out = electrons.Ek(Fwvf_out)
	#Getting the electronic dispersion for states inside Fermi surface	
	EkFwvf_in = electrons.Ek(Fwvf_in)
	#U0=8t
	U0 = 8./(N)
	omegaSat =np.max(2*(EkFwvf_out[:,len(thetaF)-1]-EkFwvf_in[:,len(thetaF)-1]))
	invGgapped = omegaSat - 2*(1-pstar1)*(EkFwvf_out[:,len(thetaF)-1]  -EkFwvf_in[:,len(thetaF)-1])-2*pstar1*(EkFwvf_out[:,len(thetaF)-1]  +EkFwvf_in[:,len(thetaF)-1])+mu1*pstar1
	invGint = invGgapped[0] - U0/2.
	lambdaEm = lambdaF[np.argmin(invGint>0)]
	Nstar = N*lambdaEm
	eGS = detEnergyAndFilling(U0, U0, mu1, Nstar,N)[0]
	fill = detEnergyAndFilling(U0, U0, mu1, Nstar,N)[1]
	print("System size, Energy per particle",N,eGS)
	return eGS
#Lattice sizes spanning from 32 x 32 to 32768 x 32768 sites
N_latt = np.array([32,64,128,256,512,1024,2048,4096,8192,16384,32768])
#No. of RG steps
RGstepL = np.array([1000,1000,1000,1000,1000,1000,1000,1000,1000,10000,10000])
energyGSperParticle = 0.0*N_latt
renV = 0.0*N_latt
params = np.asarray([process(N,RGstepL[N_latt==N]) for N in N_latt])
#plot for ground-state-energy per particle with inverse square root lattice size  
ax2.plot(1/(N_latt+0.0),params, marker='o',color='red')
ax2.set_xlabel(r"$1/\sqrt{vol}$")
ax2.set_ylabel(r"$E_{g}$")	
plt.show()	

