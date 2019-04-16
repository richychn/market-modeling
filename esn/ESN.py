#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor



def sparseRandom(s):
    '''a draw from with probability s
     of returning 0 and probability (1-s)
      of being from the standard normal distribution'''
    d=np.random.uniform(0,1)
    if d<s:
        return 0
    else:
        return np.random.randn()

def RandomMatrix(size,s,r,hermitian=False,DL=0):
    '''input a size, s for percentage of zero entries, and r for spectral radius'''
    mat=np.random.randn(size,size)
    if s>0:
        mat2=np.random.choice([0,1],size=[size,size],replace=True,p=[s,1-s])
        mat=np.multiply(mat,mat2)
    if DL!=0:
        mat=mat+DelayLine(size)*DL
    mat=np.matrix(mat)
    eigs=np.linalg.eig(mat)
    mat=mat/np.max(abs(eigs[0]))
    return mat*r

def omegaFun(eigenvectors,inH='random'):
    '''Feed in the eigenvector matrix of W,
    and this will output the omega vector defined above.
    if inH is not specified, it will generate a random h'''
    length=eigenvectors.shape[0]
    if type(inH)==np.matrixlib.defmatrix.matrix:
        h=inH
    else:
        h=[]
        for i in range(length):
            h+=[np.random.randn()]
        h=np.matrix(h).T

    omega=np.linalg.solve(eigenvectors,h)
    return omega

def ones(n):
    '''returns column vector in numpy matrix form of given length n populated by all ones'''
    return np.matrix([1]*n).T


def Sin(f,A,lam):
    '''input an autocorrelation A, lambda and f and this returns S(f) as if that was the only one'''
    
    return A*lam*((np.exp((0+1j)*f)/(1-lam*np.exp((0+1j)*f)))+np.exp((0-1j)*f)/(1-lam*np.exp((0-1j)*f)))

def S(f,A,lams):
    '''input f, array of autocorrelations and corresponding lambdas'''
    Sval=1
    for i in range(len(A)):
        Sval+=Sin(f,A[i],lams[i])
    return Sval
def S0(f):
    return (6+6*np.cos(f))/(5+2*np.cos(f))


def br(f,omega,d):
    A=omega/(np.exp((0+1j)*f)*ones(len(omega))-d)
    B=omega/(np.exp((0-1j)*f)*ones(len(omega))-d)
    return np.matmul(A,B.T)

def bI(f,omega,d,A,lams,Special=False):
    if Special==False:
        return S(f,A,lams)*br(f,omega,d)
    else:
        return S0(f)*br(f,omega,d)

def intB(omega,d,A,lams,n=5000,Special=False):
    value=bI(0,omega,d,A,lams,Special=Special)*0
    pos=-np.pi
    width=(2*np.pi)/n
    while True:
        if pos>np.pi:
            return (2*value-bI(-np.pi,omega,d,A,lams)*width-bI(np.pi,omega,d,A,lams)*width)*1/2
        else:
            value+=bI(pos,omega,d,A,lams)*width
            pos+=width

def DPart(lam1,lam2,A1,A2,d):
    left=(A1*A2)/(1-lam1*lam2)
    l1=1/((1/lam1)*ones(d.shape[0])-d)
    l2=1/((1/lam2)*ones(d.shape[0])-d)

    return left*l1*l2.T

def Dpc(lam,A,d):
    D=DPart(lam[0],lam[0],A[0],A[0],d)*0
    for i in range(len(lam)):
        for k in range(len(lam)):
            D+=DPart(lam[i],lam[k],A[i],A[k],d)
    return D

def PC(omega,lam,A,d):
    B=intB(omega,d,A,lam)
    M=np.multiply(Dpc(lam,A,d),np.linalg.inv(B))
    return 2*np.pi*omega.T*M*omega





def PCTest(size,s,r,lam,A,sym=False):
    WMatrix=RandomMatrix(size,s,r,hermitian=sym)
    omega=omegaFun(WMatrix[2])
    
    d=np.matrix(WMatrix[1]).T
    P=np.array(PC(omega,lam,A,d))[0][0]
    if P.imag>0.1:
        print('non-trivial imaginary value')
        print('imaginary part is',P.imag)
        return P.real
    else:
        return P.real

def PCComplete(W,h,lam,A):
    '''
    input:
    W as nxn np matrix
    h as nx1 np matrix
    list of lamdas
    list of A(lambdas)
    '''
    Eigs=np.linalg.eig(W)
    d=np.matrix(Eigs[0]).T
    om=omegaFun(Eigs[1],inH=h)
    PredCap=np.array(PC(om,lam,A,d))[0][0]
    if PredCap.imag>0.1:
        print('warning, imaginary part of ',PredCap.imag)
        return PredCap.real
    else:
        return PredCap.real




class LRN:
    def __init__(self,w,v):
        self.w=w
        self.v=v
        
    def GenNetState(self,TS):
        X=0*self.v
        states=[X]
        for i in range(len(TS)-1):
            X=self.w*X+self.v*TS[i]
            states+=[X]
        self.states=states
        return states
    
    def pred(self,k,TS,UseStates=False):
        if UseStates==False:
            states=self.GenNetState(TS)
        else:
            states=self.states
        if k!=0:
            states=np.array(states[:-k])
            TS=np.array(TS[k:])
        else:
            states=np.array(states)
            TS=np.array(TS)
        Pk=np.matrix((states.T*TS).mean(axis=2))
        C=np.matrix(np.mean(np.transpose(states,[0,2,1])*states,0))
        return np.array(Pk*np.linalg.inv(C)*Pk.T)[0][0]


    def mem(self,k,TS,UseStates=False):
        if UseStates==False:
            states=self.GenNetState(TS)
        else:
            states=self.states
        if k!=0:
            states=np.array(states[:-k])
            TS=np.array(TS[k:])
        else:
            states=np.array(states)
            TS=np.array(TS)
        Pk=np.matrix((states.T*TS).mean(axis=2))
        C=np.matrix(np.mean(np.transpose(states,[0,2,1])*states,0))
        return np.array(Pk*np.linalg.inv(C)*Pk.T)[0][0]

    def PC(self,dist,TS,adjusted=False,adj=False,correct=200,time=False):
        self.states=self.GenNetState(TS)

        states=self.states
        ST=np.transpose(states,[0,2,1])
        C=np.mean(ST*states,0)
        C=np.matrix(C) 
        CI=np.linalg.inv(C)
        P=0
        adjFactor=0

        
        for k in range(dist):
            if k!=0:
                states=np.array(states[:-k])
                TS=np.array(TS[k:])
            else:
                states=np.array(states)
                TS=np.array(TS)
            Pk=np.matrix((states.T*TS).mean(axis=2))
            Pr=np.array(Pk*CI*Pk.T)[0][0]
            if adjusted==True:
                adjFactor=self.w.shape[0]/(np.shape(TS)[0]-self.w.shape[0]-1)
            P+=Pr-(1-Pr)*adjFactor

        if adj==False:
            return P
        else:
            return P-self.Err(1000,1000+correct,TS)*dist


    def Err(self,a,b,TS):
        '''estimates the error and subtracts assuming farmed network states'''
        self.states=self.GenNetState(TS)
        err=[]
        for i in range(a,b):
            err+=[self.pred(i,TS,UseStates=True)]
        return np.mean(np.array(err))
    
class NRN:
    def __init__(self,w,v):
        self.w=w
        self.v=v
        
    def GenNetState(self,TS):
        X=0*self.v
        states=[X]
        for i in range(len(TS)-1):
            X=np.tanh(self.w*X+self.v*TS[i])
            states+=[X]
        self.states=states
        return states
    
    def pred(self,k,TS,UseStates=False):
        if UseStates==False:
            states=self.GenNetState(TS)
        else:
            states=self.states
        if k!=0:
            states=np.array(states[:-k])
            TS=np.array(TS[k:])
        else:
            states=np.array(states)
            TS=np.array(TS)
        Pk=np.matrix((states.T*TS).mean(axis=2))
        C=np.matrix(np.mean(np.transpose(states,[0,2,1])*states,0))
        return np.array(Pk*np.linalg.inv(C)*Pk.T)[0][0]

    def mem(self,k,TS,UseStates=False):
        if UseStates==False:
            states=self.GenNetState(TS)
        else:
            states=self.states
        if k!=0:
            states=np.array(states[:-k])
            TS=np.array(TS[k:])
        else:
            states=np.array(states)
            TS=np.array(TS)
        Pk=np.matrix((states.T*TS).mean(axis=2))
        C=np.matrix(np.mean(np.transpose(states,[0,2,1])*states,0))
        return np.array(Pk*np.linalg.inv(C)*Pk.T)[0][0]
    def PC(self,dist,TS,adj=False,correct=1000):
        self.states=self.GenNetState(TS)
        P=0
        for i in range(dist):
            P+=self.pred(i,TS,UseStates=True)
        if adj==False:
            return P
        else:
            return P-self.Err(100,100+correct,TS)*dist
    def Err(self,a,b,TS):
        '''estimates the error and subtracts assuming farmed network states'''
        self.states=self.GenNetState(TS)
        err=[]
        for i in range(a,b):
            err+=[self.pred(i,TS,UseStates=True)]
        return np.mean(np.array(err)) 
    def MC(self,dist,TS):
        self.states=self.GenNetState(TS)
        P=0
        for i in range(1,dist):
            P+=self.mem(i,TS,UseStates=True)
        return P
    


def EvenProcess(l):
    '''generates a sequence of length l according the the even process defined by Dr. Marzen in "A Difference Between Memory
    and Prediction'''
    states=['a','b']
    state=states[np.random.randint(0,2)]
    seq=[]
    aOuts=[-1*np.sqrt(2),1/np.sqrt(2)]
    
    for i in range(l):
        if state=='b':
            seq+=[1/np.sqrt(2)]
            state='a'
        elif state=='a':
            rn=np.random.randint(0,2)
            seq+=[aOuts[rn]]
            state=states[rn]
    return seq

def HeavySidedMakeMats(states,signals):
    T=np.random.dirichlet([1]*states,states).T
    sp=np.random.uniform(-30,30,size=[states,states])
    sL=(np.tanh(sp)+1)/2
    sL2=1-sL
    sL=sL*T
    sL2=sL2*T
    return np.array([sL,sL2])

def MakeMats(states,signals,LTDiscrete=True):
    '''
    input:
        number of states
        number of different signals
    ouput:
        labeled transition matrices'''
    T=np.random.dirichlet([1]*states,states).T
    if LTDiscrete==False:
        Sp=np.random.dirichlet([1]*signals,[states,states]).T
    else:
        Sp=np.random.multinomial(1,[1/(signals)]*signals,size=states**2).T.reshape([signals,states,states])
    LT=T*Sp
    return LT

def RandomUniMat(n,r):
    D=np.matrix(np.diagflat(np.random.uniform(-r,r,size=n)))
    P=RandomMatrix(n,0,1)
    return P*D*np.linalg.inv(P)

#props=np.array([sum(out==Outputs[0]),sum(out==Outputs[1]),sum(out==Outputs[2])])/n

def GenHMM(LT,outs,n,init='random',norm=True):
    '''
    input:
        Labeled transition matrices and labels
        n steps
    output:
        array of the time series generated by given HMM
        '''
    T=np.sum(LT,0)
    Sp = np.divide(LT, T, out=np.zeros_like(LT), where=T!=0)
    Outputs=outs
    EIGEN=np.linalg.eig(T)
    
    pos=np.where(EIGEN[0]==np.max(EIGEN[0]))[0]
    StatDis=(EIGEN[1].T[pos]/(np.sum(EIGEN[1].T[pos])))[0].real
    if norm==True:
        SigDis=np.dot(np.sum(LT,1),StatDis)
        Outputs=Outputs-(np.dot(SigDis,Outputs))
        Outputs=Outputs/np.sqrt((np.dot(SigDis,Outputs**2)))
    
    if init=='random':
        oldS=np.dot(np.random.multinomial(1,StatDis),np.array(range(T.shape[0])))
    else:
        oldS=init
    out=[]
    
    for i in range(n):
        newS=np.dot(np.random.multinomial(1,T.T[oldS]),np.array(range(T.shape[0])))
        out+=[np.dot(np.random.multinomial(1,Sp.T[oldS][newS]),Outputs)]
        oldS=newS
    out=np.array(out)
    
    return out


def CondNum(mat):
    SD=np.linalg.svd(mat)[1]
    return np.max(SD)/np.min(SD)

def AutoCorr(LT,outs,ConditionCheck=False):
    '''
    input:
        LT=labeled transition matrices
        outs=labels
        outputs a list of A(lam) and lambdas
    '''
    n=LT[0].shape[0]
    nOut=len(outs)
    T=np.sum(LT,0)


    Sp = np.divide(LT, T, out=np.zeros_like(LT), where=T!=0)
    Outputs=outs
    S=np.matrix(np.sum(LT*outs.reshape([nOut,1,1]),0))
    EIG=np.linalg.eig(T)
    D=np.diag(EIG[0])
    P=EIG[1]
    L=ones(n).T*S*P*np.linalg.inv(D)
    pos=np.where(np.linalg.eig(T)[0]==np.max(abs(np.linalg.eig(T)[0])))
    peq=EIG[1].T[pos]
    peq=peq/np.sum(peq)
    R=(np.linalg.inv(P)*S*peq.T).T

    A=np.array(np.multiply(L,R))[0]
    lam=EIG[0]

    idx = lam.argsort()[::-1]  
    A = A[idx]
    lam = lam[idx]
    one=np.where(lam==np.max(abs(lam)))
    A[one]=0
    lam[one]=1
    A=A[1:]
    lam=lam[1:]
    return [A,lam]

def MatPC(W,h,LT,outs):
    '''
    input:
        reservoir properties W and h
        Labeled transition matrices and labels
    output:
        PC as defined in paper's formula
    '''
    AC=AutoCorr(LT,outs)

    return PCComplete(W,h,AC[1],AC[0])


def UnifSigs(LT):
    T=np.sum(LT,0)
    EIGEN=np.linalg.eig(T)
    pos=np.where(EIGEN[0]==np.max(EIGEN[0]))[0]

    Outputs=np.random.uniform(-1,1,LT.shape[0])
    
    StatDis=(EIGEN[1].T[pos]/(np.sum(EIGEN[1].T[pos])))[0].real
    
    SigDis=np.dot(np.sum(LT,1),StatDis)
    
    Outputs=Outputs-(np.dot(SigDis,Outputs))
    
    Outputs=Outputs/np.sqrt((np.dot(SigDis,Outputs**2)))
    
    return Outputs

def hmake(n,n_0=0,lower=-1,upper=1):
    vec=np.array([0]*n_0+(n-n_0)*[1])
    hzero=np.random.permutation(vec)
    h=np.random.uniform(lower,upper,n)
    h=np.multiply(h,hzero)
    h=np.matrix(h).T
    return h


def Rss(t,A,lam):
    A=np.array(A)
    lam=np.array(lam)
    if t==0:
        return 1
    
    return np.sum(A*(lam**t))

def Pk(tau,A,lam,dist):
    p=[]
    for i in range(tau,tau+dist):
        p+=[Rss(i,A,lam)]
    return np.matrix(p).T

def linPCmax(A,lam,size,dist):
    R=[[Rss(abs(i-k),A,lam) for k in range(size)] for i in range(size)]
            
    R=np.matrix(R).reshape([size,size])
    Rinv=np.linalg.inv(R)
    PC=0
    for i in range(dist):
        p=Pk(i,A,lam,size)
        PC+=p.T*Rinv*p
    
    return (np.array(PC)[0][0]-1).real


def linMemMax(A,lam,k,size=20):
    R=[[Rss(abs(i-k),A,lam) for k in range(size)] for i in range(size)]
            
    R=np.matrix(R).reshape([size,size])
    
    p=Pk(k,A,lam,size)
    return (p.T*np.linalg.inv(R)*p).A1[0]

def DelayLine(n):
    W=np.concatenate((np.zeros(n-1).reshape(1,n-1),np.identity(n-1)))
    W=np.concatenate((W,np.zeros(n).reshape(n,1)),1)
    return W

from sklearn import svm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor



class ESN:
    
    def __init__(self,w,v,X='None',Activation=np.tanh,alpha=1,beta=0,PCA=False,n_components=100,UseLinear=True,readout=None):
        self.w=w
        self.v=v
        self.X=X
        self.Activation=Activation
        self.alpha=alpha
        self.beta=beta
        self.PCA=PCA
        self.n_components=n_components
        self.states=[]
        if UseLinear==False:
            self.readout=readout
        self.UseLinear=UseLinear
        self.n=v.shape[0]
        
    def GenNetState(self,TS,noise=True):
        if type(self.X)==str:
            self.X=np.matrix([0]*self.w.shape[0]).reshape((self.w.shape[0],1))
        states=[]
        for i in range(len(TS)):
            self.X=self.Activation(np.dot(self.w,self.X)+np.dot(self.v,TS[i]))
            states+=[self.X]
            self.states+=[self.X]
        return np.matrix(np.transpose(np.array(states),[2,0,1])[0])
    
    def fit(self,inTS,out,score=False):
        '''input a sequence X and a target sequence y to fit weights'''
        states = self.GenNetState(inTS)
        if self.UseLinear==True:
            if self.alpha==0 and self.beta==0:
                self.readout=linear_model.LinearRegression()
                
            elif self.alpha==0:
                self.readout = linear_model.Lasso(alpha=self.beta)
                
            elif self.beta==0:
                self.readout = linear_model.Ridge(alpha=self.alpha)
                
            else:
                self.readout = linear_model.ElasticNet(alpha=self.alpha,l1_ratio=self.beta/self.alpha)
            
        if self.PCA==True:
            self.meansub=states.mean(0).reshape(1,self.n)
            states=states-self.meansub
            U,sigma,W=np.linalg.svd(states)
            self.Wtrunk=W.T[:self.n_components].T
            states=states*self.Wtrunk
            
        self.readout.fit(states,out)
        
        if self.UseLinear==True:
            self.Weights=np.matrix(self.readout.coef_).T
            self.intercept=self.readout.intercept_

        if score==True:
            return [self.Weights,self.readout.score(states,out)]


    def pca(self,X):
        if self.PCA==False:
            return X
        else:
            return (X-self.meansub)*self.Wtrunk
    
    def kStepFit(self,TS,k,score=False):
        if score==False:
            self.fit(TS[:-k],TS[k:],score=score)
        else:
            return self.fit(TS[:-k],TS[k:],score=score)[1]
        
    def predict(self,inTS):
        states=self.GenNetState(inTS)
        states = self.pca(states)
        
        return self.readout.predict(states)
    
    def FreeRun(self,length):
        states=[]
        for i in range(length):
            pred=self.readout.predict(self.pca(self.X.T))[0]
            self.X=self.Activation(np.dot(self.w,self.X)+np.dot(self.v,pred))
            states+=[self.X]
        stateMat=np.transpose(np.array(states),[2,0,1])[0]
        preds=self.readout.predict(self.pca(stateMat))
        return preds


def meanPred(prediction):
    return np.mean(prediction,0)

class Ensemble(BaseEstimator, ClassifierMixin):  
    """An example of classifier"""

    def __init__(self, estimators=[svm.SVR(kernel='linear'),svm.SVR(kernel='rbf')],fitter='mean',ReadoutVal=False):
        """
        Called when initializing the classifier
        """
        self.estimators=estimators
        self.fitter=fitter
        self.ReadoutVal=ReadoutVal



    def fit(self, X, y):        
        if self.ReadoutVal==True:
            permute=np.random.permutation(y.shape[0])

            X=X[permute]
            y=y[permute]
            split=17*(y.shape[0]//20)

            Xtrain=X[:split]
            ytrain=y[:split]

            Xtest=X[split:]
            ytest=y[split:]

            for model in self.estimators:
                
                model.fit(Xtrain,ytrain)

            testPreds=[]
            for model in self.estimators:
                testPreds+=[model.predict(Xtest)]
            testPreds=np.array(testPreds).T
            self.fitter.fit(testPreds,ytest)


        else:
            for model in self.estimators:
                
                model.fit(X,y)

            testPreds=[]

            if self.fitter=='mean':
                return

            else:
                for model in self.estimators:
                    testPreds+=[model.predict(X)]
                
                testPreds=np.array(testPreds).T
                self.fitter.fit(testPreds,y)
        

    def predict(self, X):
        predictions=[]
        for model in self.estimators:
            predictions+=[model.predict(X)]
        predictions=np.array(predictions).T
        
        return self.fitter.predict(predictions)

    def score(self, X, y):
        preds=self.predict(X)
        
        return np.mean((preds-y)**2)


# In[3]:


def Identity(X):
    return X

def LeakyTanh(X):
    return np.tanh(2*X)+X/5

def splitAct(x):
    x[::2]=x[::2]
    x[1::2]=np.tanh(x[1::2])
    return x

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


# In[4]:


class ESN:
    
    def __init__(self,w,v,X='None',Activation=np.tanh,alpha=1,beta=0,PCA=False,n_components=100,UseLinear=True,readout=None):
        self.w=w
        self.v=v
        self.X=X
        self.Activation=Activation
        self.alpha=alpha
        self.beta=beta
        self.PCA=PCA
        self.n_components=n_components
        self.states=[]
        if UseLinear==False:
            self.readout=readout
        self.UseLinear=UseLinear
        self.n=v.shape[0]
        
    def GenNetState(self,TS,noise=True):
        if type(self.X)==str:
            self.X=np.matrix([0]*self.w.shape[0]).reshape((self.w.shape[0],1))
        states=[]
        for i in range(len(TS)):
            self.X=self.Activation(np.dot(self.w,self.X)+np.dot(self.v,TS[i]))
            states+=[self.X]
            self.states+=[self.X]
        return np.matrix(np.transpose(np.array(states),[2,0,1])[0])
    
    def fit(self,inTS,out,score=False):
        '''input a sequence X and a target sequence y to fit weights'''
        states = self.GenNetState(inTS)
        if self.UseLinear==True:
            if self.alpha==0 and self.beta==0:
                self.readout=linear_model.LinearRegression()
                
            elif self.alpha==0:
                self.readout = linear_model.Lasso(alpha=self.beta)
                
            elif self.beta==0:
                self.readout = linear_model.Ridge(alpha=self.alpha)
                
            else:
                self.readout = linear_model.ElasticNet(alpha=self.alpha,l1_ratio=self.beta/self.alpha)
            
        if self.PCA==True:
            self.meansub=states.mean(0).reshape(1,self.n)
            states=states-self.meansub
            U,sigma,W=np.linalg.svd(states)
            self.Wtrunk=W.T[:self.n_components].T
            states=states*self.Wtrunk
            
        self.readout.fit(states,out)
        
        if self.UseLinear==True:
            self.Weights=np.matrix(self.readout.coef_).T
            self.intercept=self.readout.intercept_

        if score==True:
            return [self.Weights,self.readout.score(states,out)]


    def pca(self,X):
        if self.PCA==False:
            return X
        else:
            return (X-self.meansub)*self.Wtrunk
    
    def kStepFit(self,TS,k,score=False):
        if score==False:
            self.fit(TS[:-k],TS[k:],score=score)
        else:
            return self.fit(TS[:-k],TS[k:],score=score)[1]
        
    def predict(self,inTS):
        states=self.GenNetState(inTS)
        states = self.pca(states)
        
        return self.readout.predict(states)
    
    def FreeRun(self,length):
        states=[]
        for i in range(length):
            pred=self.readout.predict(self.pca(self.X.T))[0]
            self.X=self.Activation(np.dot(self.w,self.X)+np.dot(self.v,pred))
            states+=[self.X]
        stateMat=np.transpose(np.array(states),[2,0,1])[0]
        preds=self.readout.predict(self.pca(stateMat))
        return preds



# In[5]:


TS=[]
for i in range(1000):
    TS+=[np.random.uniform(-1,1,1)]
TS=np.array(TS)
out=[0]*10
for i in range(10,1000):
    out+=[np.sum(TS[i-10:i])]


n=300
W=RandomMatrix(n,0,0.9)
v=np.matrix(np.random.uniform(-1,1,size=n)).T


TSTrain=TS[:800]
TSTest=TS[800:]

Net=ESN(W,v,Activation=Identity)

Net.fit(TSTrain,out[:800])
preds=Net.predict(TSTest)


# In[6]:


def uniEIGMat(n,r,sparsity=0):
    P=np.matrix(np.random.uniform(-1,1,[n,n]))
    PI=np.linalg.inv(P)
    D=np.diag(np.random.uniform(-r,r,size=n))
    return P*D*PI


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

def partition(data,mem_size=100):
    '''takes in a timeseries and returns an array of mem_size columns, length of the data-n_steps rows,
    ready for prediction in weiner filter'''
    for i in range(len(data)-mem_size-1):
        A=data[i:i+mem_size]
        if i==0:
            SubArray=A
        else:
            SubArray=np.vstack((SubArray,A))
    return SubArray

def genOutput(data,mem_size=100):
    return data[mem_size+1:]

class WeinerFilter:
    def __init__(self,mem_size=50,Type='Logit'):
        self.mem_size=mem_size
        if Type=='Logit':
            self.model=LogisticRegression()
        else:
            self.model=Type
        
    def fit(self,data):
        X=partition(data,self.mem_size)
        y=genOutput(data,self.mem_size)
        self.model.fit(X,y)
        
    def predict(self,data,proba=False):
        '''this performs 1 step prediction, not free-running prediction'''
        X=partition(data,self.mem_size)
        y=genOutput(data,self.mem_size)
        if proba==False:
            predictions=self.model.predict(X)
            print('Mean Squared Error is', np.mean((predictions-y)**2))
        else:
            predictions=self.model.predict_proba(X)[:,1]
            print('Mean Squared Error is', np.mean((predictions-y)**2))
        return predictions
    def error(self,data,proba=False):
        X=partition(data,self.mem_size)
        y=genOutput(data,self.mem_size)
        if proba==False:
            predictions=self.model.predict(X)
            MSE=np.mean((predictions-y)**2)
        else:
            predictions=self.model.predict_proba(X)[:,1]
            MSE=np.mean((predictions-y)**2)
        return MSE


# In[22]:


def DelayLine(n):
    W=np.concatenate((np.zeros(n).reshape(1,n),np.identity(n)))
    W=np.concatenate((W,np.zeros(n+1).reshape(n+1,1)),1)
    return W

