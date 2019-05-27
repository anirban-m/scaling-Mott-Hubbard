# scaling-Mott-Hubbard
Via a Hamiltonian renormalization group study for the electronic unfrustrated Hubbard model on a 2d square lattice 
we obtain the ground state energy, doping as a function of effective chemical potential. Below we list the contents
of this github repo.

1>GS.py creates a plot of ground state energy per particle with increasing lattice sizes (N=256x256, 
    512x512,1024x1024,2048x2048,4096x4096,8192x8192,16384x16384,32768x32768) at U0=8t with t=1 for the
    half filled Hubbard model.

2>GS-dopedML.py creates plots for: 
   (a) Variation of doping fraction with effective  chemical potential for system size 1024x1024 at U0=8t, t=1.
   (b)Variation of ground state energy  with doping for system sizes 1024x1024 at U0=8t, t=1. \
   (c) A plot of ground state energy per particle with increasing lattice sizes (N=32x32,64x64,128x128, 256x256, 
         512x512,1024x1024,2048x2048,4096x4096,8192x8192,16384x16384,32768x32768) at U0=8t, t=1,
         f_h = 0.125(doping fraction).
