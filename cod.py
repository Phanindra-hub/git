import pandas as pd
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as si
import random
import math
#%%
data = pd.DataFrame()
#%%
import datetime as dt
start = dt.datetime(2019, 1, 1)
end = dt.datetime(2021, 12, 31)
#%%
import pandas_datareader.data as web
# Stocks from Yahoo finance
Apple = web.DataReader('AAPL', 'yahoo', start, end)['Adj Close']

#%%
#liabilities=[255827(dec29 18),236138(mar30 19),225783(jun30 19),248028(sep30 19),251087(dec30 19),241975(mar28 20),#245062(jun27 20),258549(sep26 20),287830(Dec26 20),267980(mar30 21),265560(jun30 21),287912(sep30 21)]
#shares_outstanding=[3083.7562124(jan1-mar30 19),2289.35986159(mar30-jun30 19),1994.54094293(jun30-sep30 19),1686.00708031(sep30-dec30 19),1255.69424965(dec30-mar30 20),1283.55(mar30-jun30 20),826.552315609(jun30-sep30 20),587.264066151(sep30-dec30 20),505.565310329(dec30-mar30 21),574.138932692(mar30-jun30 21),484.94907582(jun30-sep30 21),435.193488308(sep30-dec30 21)]
#%%
lis = []
for i in range(63):
    lis.append(255827)
for i in range(63):
    lis.append(236138) 
for i in range(63):
    lis.append(225783) 
for i in range(63):
    lis.append(248028) 
for i in range(63):
    lis.append(251087) 
for i in range(63):
    lis.append(241975) 
for i in range(63):
    lis.append(245062) 
for i in range(63):
    lis.append(258549) 
for i in range(63):
    lis.append(287830) 
for i in range(63):
    lis.append(267980)
for i in range(63):
    lis.append(265560) 
for i in range(63):
    lis.append(287912) 
liabilities = lis
#%%
nos = []
for i in range(63):
    nos.append(3083.7562124)
for i in range(63):
    nos.append(2289.35986159) 
for i in range(63):
    nos.append(1994.54094293) 
for i in range(63):
    nos.append(1686.00708031) 
for i in range(63):
    nos.append(1255.69424965) 
for i in range(63):
    nos.append(1283.55) 
for i in range(63):
    nos.append(826.552315609) 
for i in range(63):
    nos.append(587.264066151) 
for i in range(63):
    nos.append(505.565310329) 
for i in range(63):
    nos.append(574.138932692)
for i in range(63):
    nos.append(484.94907582) 
for i in range(63):
    nos.append(435.193488308) 
const = nos
#%%
# Risk-free rate ( One month US Treasury bill from  Fama and Kenneth R. French.)
rf = pd.read_csv('/Users/phanidodda/Desktop/Spring 2022/MF 825 Portfolio Theory/FF-3Fac-day.csv')
#%%
rff=[]
for i in range(24391,25147):
    rff.append(rf.iloc[i,4])
# rf = rf.tail(319).head(260)
# rf = rf.RF

#%%
data = pd.DataFrame([Apple]).T
data = data.iloc[:-1,:]
data.columns = ['Equity']
#%%
data['Liabilities'] =lis
data['Liabilities']=data['Liabilities']/1000
data['const']=const
data['Risk_free'] = rff
data = data[['Equity','Liabilities','const','Risk_free']]
#%%
data['Equity']=data['Equity']*data['const']
#%%
data['Assets']=(data['Equity']/1000+data['Liabilities'])
#%%
data['Log returns'] = np.log(data['Assets']/data['Assets'].shift())

#%%

#%%
n=len(data)
itr=5000
r=rff
phi=0
sigma=data['Log returns'].std()*252**.5
#%%

lambdaq = 0.01
sigmapi = 0.5
upi=0
vq = math.exp(upi + (sigmapi**2)/2) - 1
#sigma = np.sqrt(0.035- lambdaq*((sigmapi)**2))
#%%
ux=[]
for i in range(n):
    ux.append((r[i] - phi - ((sigma)**2)/2 - lambdaq*vq)*10/n)
#%%    
T=3
sigmax = sigma*np.sqrt(T/n)
#%%

V=np.zeros((itr,n))
#%%
X=data['Assets']/data['Liabilities']
V[:,0] = X[0]
prob_zero = 1 - (lambdaq * (T/n))
prob_one = lambdaq * (T/n)
#V[:,1] = 0
#%%
k=[]
for j in range(itr):
    W=0
    for i in range(1,n):
        x=np.random.normal(ux[i], sigmax)
        pi=np.random.normal(upi, sigmapi)
        y = int(np.random.choice(2, 1, p = [prob_zero, prob_one]))
        V[j,i]=math.exp(np.log(V[j,i-1])+x+(pi*y))
        if(V[j,i]<1):
            W=W+1.3-V[j,i]
            break
    k.append(W/itr)
#%%
B=np.zeros(n)
#B[0]=1
#r=0.03
R=0.37
prob=[]
for i in range(0,n):
    a=0
    b=0
    B[i]=np.exp(-T*r[i]*i/n)*(1-sum(k))
#%%
plt.plot(B)
#%%
#if risk-free rate is constant
# B=np.zeros(n)
# #B[0]=1
# r=0.03
# prob=[]
# for i in range(0,n):
#     a=0
#     b=0
#     B[i]=np.exp(-T*r*i/n)*(1-sum(k))
# plt.plot(B)
    
#%%
#credit Default Spreads
R=0.37
probq=[B[0]]
rates=[rff[0]]
l=range(0,int(len(B)/63))
#%%
for i in range(len(l)):
    if (i==len(l)-1):
        probq.append(B[-1])
        rates.append(rff[-1])
        break
    else:
        probq.append(B[(i+1)*63])
        rates.append(rff[(i+1)*63])
#%%
spread=[]
rpv=[]
for i in range(1,len(rates)):
    rpv01=0
    num=0
    for j in range(1,i):
        rpv01=rpv01+(0.25*math.exp((-rates[j])*(j*63/len(B)))*(probq[j-1]+probq[j]))
        num=num+(math.exp((-rates[j])*(i*63/len(B)))*(probq[j-1]-probq[j]))
    if(i!=1):
        
        spread.append(((1-R)*num)/rpv01)
#%%
data_spread = pd.DataFrame()
data_spread['spread']=spread
plt.plot(abs(data_spread.pct_change()))   





        
    
