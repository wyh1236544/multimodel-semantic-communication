"""N648/Z27 QC-LDPC numeric tables; independently written NumPy encoder/decoder.

R1/2: J. Milos DREL FEEC BUT LDPC_matrix.m, IEEE802.11n/ac table.
R3/4: MathWorks ldpcEncode IEEE802.11 example. See PROTOCOL.md URLs.
Right-shifted identity: H[row*z+i,col*z+(i+shift)%z]=1.
Not a complete WLAN PHY and not a capacity-achieving-decoder claim.
"""
from functools import lru_cache
import numpy as np

BASE12=[
[0,-1,-1,-1,0,0,-1,-1,0,-1,-1,0,1,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
[22,0,-1,-1,17,-1,0,0,12,-1,-1,-1,-1,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1],
[6,-1,0,-1,10,-1,-1,-1,24,-1,0,-1,-1,-1,0,0,-1,-1,-1,-1,-1,-1,-1,-1],
[2,-1,-1,0,20,-1,-1,-1,25,0,-1,-1,-1,-1,-1,0,0,-1,-1,-1,-1,-1,-1,-1],
[23,-1,-1,-1,3,-1,-1,-1,0,-1,9,11,-1,-1,-1,-1,0,0,-1,-1,-1,-1,-1,-1],
[24,-1,23,1,17,-1,3,-1,10,-1,-1,-1,-1,-1,-1,-1,-1,0,0,-1,-1,-1,-1,-1],
[25,-1,-1,-1,8,-1,-1,-1,7,18,-1,-1,0,-1,-1,-1,-1,-1,0,0,-1,-1,-1,-1],
[13,24,-1,-1,0,-1,8,-1,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,-1,-1,-1],
[7,20,-1,16,22,10,-1,-1,23,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,-1,-1],
[11,-1,-1,-1,19,-1,-1,-1,13,-1,3,17,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,-1],
[25,-1,8,-1,23,18,-1,14,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0],
[3,-1,-1,-1,16,-1,-1,2,25,5,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0]]
BASE34=[
[16,17,22,24,9,3,14,-1,4,2,7,-1,26,-1,2,-1,21,-1,1,0,-1,-1,-1,-1],
[25,12,12,3,3,26,6,21,-1,15,22,-1,15,-1,4,-1,-1,16,-1,0,0,-1,-1,-1],
[25,18,26,16,22,23,9,-1,0,-1,4,-1,4,-1,8,23,11,-1,-1,-1,0,0,-1,-1],
[9,7,0,1,17,-1,-1,7,3,-1,3,23,-1,16,-1,-1,21,-1,0,-1,-1,0,0,-1],
[24,5,26,7,1,-1,-1,15,24,15,-1,8,-1,13,-1,13,-1,11,-1,-1,-1,-1,0,0],
[2,2,19,14,24,1,15,19,-1,21,-1,2,-1,24,-1,3,-1,2,1,-1,-1,-1,-1,0]]

def inverse(a):
    a=np.asarray(a,dtype=np.uint8);n=len(a)
    m=np.concatenate((a.copy(),np.eye(n,dtype=np.uint8)),axis=1)
    for col in range(n):
        piv=np.flatnonzero(m[col:,col])
        if not len(piv):raise ValueError('Singular LDPC parity partition')
        j=col+int(piv[0]);m[[col,j]]=m[[j,col]]
        rows=np.flatnonzero(m[:,col]);rows=rows[rows!=col];m[rows]^=m[col]
    return m[:,n:]

class Code:
    def __init__(self,rate):
        if rate not in ('12','34'):raise ValueError('Unsupported standard table')
        base=np.array(BASE12 if rate=='12' else BASE34,dtype=int)
        self.n=648;self.m=len(base)*27;self.k=self.n-self.m
        self.h=np.zeros((self.m,self.n),dtype=np.uint8)
        for r,c in zip(*np.where(base>=0)):
            self.h[r*27+np.arange(27),c*27+(np.arange(27)+base[r,c])%27]=1
        b=inverse(self.h[:,self.k:]).astype(np.uint16)
        self.generator=((b@self.h[:,:self.k].astype(np.uint16))%2).astype(np.uint16)
        deg=self.h.sum(1);self.mask=np.arange(int(deg.max()))[None,:]<deg[:,None]
        self.adj=np.full(self.mask.shape,self.n,dtype=int)
        for i,row in enumerate(self.h):self.adj[i,:int(deg[i])]=np.flatnonzero(row)
    def encode(self,info):
        x=np.asarray(info,dtype=np.uint8)
        if x.ndim!=2 or x.shape[1]!=self.k or ((x!=0)&(x!=1)).any():raise ValueError('LDPC info geometry')
        p=(x.astype(np.uint16)@self.generator.T)%2
        out=np.concatenate((x,p.astype(np.uint8)),axis=1)
        if self.syndrome(out).any():raise RuntimeError('Internal LDPC encoding failed')
        return out
    def syndrome(self,bits):
        x=np.pad(np.asarray(bits,dtype=np.uint8),((0,0),(0,1)))
        return np.bitwise_xor.reduce(x[:,self.adj],axis=2)
    def decode(self,llr,iterations=40,alpha=.8,known_zero_tail=0):
        ch=np.clip(np.asarray(llr,dtype=float),-50,50).copy()
        if ch.ndim!=2 or ch.shape[1]!=self.n or not np.isfinite(ch).all():raise ValueError('LDPC LLR geometry')
        if not 1<=iterations<=100 or not 0<alpha<=1 or not 0<=known_zero_tail<=self.k:raise ValueError('Decoder bounds')
        if known_zero_tail:ch[-1,self.k-known_zero_tail:self.k]=50
        batch=len(ch);belief=np.pad(ch,((0,0),(0,1)));messages=np.zeros((batch,)+self.adj.shape)
        found=np.zeros(batch,dtype=bool);done=np.zeros(batch,dtype=int);answer=np.zeros((batch,self.n),dtype=np.uint8)
        offsets=np.arange(batch)[:,None,None]*(self.n+1);indices=(offsets+self.adj[None]).ravel()
        for it in range(1,iterations+1):
            q=belief[:,self.adj]-messages
            aq=np.where(self.mask[None],np.abs(q),np.inf)
            first=np.argmin(aq,axis=2);two=np.partition(aq,1,axis=2)[:,:,:2]
            lo=two[:,:,0];hi=two[:,:,1]
            mag=np.where(np.arange(aq.shape[2])[None,None,:]==first[:,:,None],hi[:,:,None],lo[:,:,None])
            sign=np.where(q<0,-1.,1.);sign=np.where(self.mask[None],sign,1.)
            messages=np.where(self.mask[None],alpha*sign.prod(axis=2)[:,:,None]*sign*mag,0.)
            sums=np.bincount(indices,weights=messages.ravel(),minlength=batch*(self.n+1)).reshape(batch,self.n+1)
            belief[:,:self.n]=np.clip(ch+sums[:,:self.n],-100,100);belief[:,self.n]=0
            hard=(belief[:,:self.n]<0).astype(np.uint8)
            valid=~self.syndrome(hard).any(axis=1);new=valid&~found
            answer[new]=hard[new];done[new]=it;found|=new
            if found.all():break
        answer[~found]=hard[~found];done[~found]=it
        return answer[:,:self.k],dict(converged=found.tolist(),iterations=done.tolist())

@lru_cache(maxsize=2)
def code(rate):return Code(rate)
