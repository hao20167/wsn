# --- vvs_hco.py ---
import numpy as np
class VVSHCO_Optimizer:
    def __init__(self,obj_fn,bounds,pop_size=50,max_iters=200):
        self.obj_fn=obj_fn; self.bounds=np.array(bounds)
        self.pop=pop_size; self.dim=self.bounds.shape[0]; self.max_iters=max_iters
    def optimize(self):
        X=np.random.uniform(self.bounds[:,0],self.bounds[:,1],(self.pop,self.dim))
        Xopp=self.bounds[:,0]+self.bounds[:,1]-X
        F=np.apply_along_axis(self.obj_fn,1,X)
        Fopp=np.apply_along_axis(self.obj_fn,1,Xopp)
        F=np.nan_to_num(F,nan=np.inf); Fopp=np.nan_to_num(Fopp,nan=np.inf)
        mask=F>Fopp; X[mask]=Xopp[mask]; F[mask]=Fopp[mask]
        pbest_pos=X.copy(); pbest_val=F.copy()
        g_idx=np.argmin(pbest_val); gbest_pos=pbest_pos[g_idx].copy()
        wg1_min,wg1_max=0.4,0.9; gamma,eta=0.5,0.5
        Ra=0.1*np.sqrt(np.sum((self.bounds[:,1]-self.bounds[:,0])**2)/self.pop)
        velocities=np.zeros_like(X)
        for k in range(self.max_iters):
            jc=k+1; c1=c2=1+1.5*(1-np.exp(-jc/600.0))
            wg1=wg1_min+(k/self.max_iters)*(wg1_max-wg1_min)
            k_best=int(self.pop-(self.pop-2)*np.sqrt(jc/self.max_iters))
            idx_sorted=np.argsort(pbest_val)
            Initialbest=pbest_pos[idx_sorted[:k_best]].mean(axis=0)
            favg=np.nanmean(pbest_val); fbest=np.nanmin(pbest_val)
            denom=4*eta*(fbest-favg)+1e-9
            eps=gamma*((fbest-pbest_val)**2-(favg-pbest_val)**2)/denom
            eps=np.nan_to_num(eps,nan=0.0,posinf=0.0,neginf=0.0)
            for j in range(self.pop):
                b1=pbest_pos[j]-X[j]; b2=gbest_pos-X[j]
                v=(wg1*(velocities[j]+eps[j])
                   +c1*b1*np.sin(2*np.pi*jc/self.max_iters)
                   +c2*b2*np.sin(2*np.pi*jc/self.max_iters)
                   +(1-0.1*jc)*(Initialbest-(X[j]+(1-0.1*jc)*Ra*np.random.uniform(-1,1,self.dim))))
                velocities[j]=v; X[j]=np.clip(X[j]+v,self.bounds[:,0],self.bounds[:,1])
                Fv=self.obj_fn(X[j])
                if Fv<pbest_val[j]: pbest_val[j]=Fv; pbest_pos[j]=X[j].copy()
            g_idx=np.argmin(pbest_val); gbest_pos=pbest_pos[g_idx].copy()
        return gbest_pos,pbest_val[g_idx]