# --- dpso.py ---
import numpy as np
class DPSO_Optimizer:
    def __init__(self,obj_fn,bounds,pop_size=50,max_iters=200):
        self.obj_fn=obj_fn; self.bounds=np.array(bounds)
        self.pop=pop_size; self.dim=self.bounds.shape[0]; self.max_iters=max_iters
    def optimize(self):
        X=np.random.uniform(self.bounds[:,0],self.bounds[:,1],(self.pop,self.dim))
        V=np.zeros_like(X)
        pbest_pos=X.copy(); pbest_val=np.apply_along_axis(self.obj_fn,1,X)
        g_idx=np.argmin(pbest_val); gbest_pos=pbest_pos[g_idx].copy()
        w,c1,c2=0.7,1.4,1.4
        for _ in range(self.max_iters):
            for j in range(self.pop):
                r1,r2=np.random.rand(),np.random.rand()
                V[j]=w*V[j]+c1*r1*(pbest_pos[j]-X[j])+c2*r2*(gbest_pos-X[j])
                X[j]=np.clip(X[j]+V[j],self.bounds[:,0],self.bounds[:,1])
                fv=self.obj_fn(X[j])
                if fv<pbest_val[j]: pbest_val[j]=fv; pbest_pos[j]=X[j].copy()
            g_idx=np.argmin(pbest_val); gbest_pos=pbest_pos[g_idx].copy()
        return gbest_pos,pbest_val[g_idx]