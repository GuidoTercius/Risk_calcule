#############################################################################
################################# Importing #################################
#############################################################################

from matplotlib import dates as mpl_date
from datetime import datetime,timedelta 
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import pandas as pd
from  math import *
import numpy as np
import sys


plt.style.use('seaborn') 

#############################################################################
############################## Metrics of risk ##############################
#############################################################################

start = "1900-02-06"
end= "2023-03-13"       
stock = "BEEF3.SA"
s1=yf.download(stock , start=start )['Adj Close'].ffill()   


#############################################################################
############################# Moving Average ################################
#############################################################################


L= (ceil(len(s1)/2.5))
M= (ceil(len(s1)/5))
S= (ceil(len(s1)/10))
M_L=s1.rolling(L).mean()
M_M=s1.rolling(M).mean()
M_S=s1.rolling(S).mean()
SMm=pd.DataFrame()
SMm['Average Moving '+ str(L) + ' days']=M_L
SMm['Average Moving '+ str(M) + ' days']=M_M
SMm['Average Moving '+ str(S) + ' days']=M_S
SMm['Price']=s1


SMm.plot()
plt.title('Price Change of: ' + stock)
plt.xlabel('Date')
plt.ylabel('Price')

plt.gcf().autofmt_xdate()
date_format=mpl_date.DateFormatter('%d/%m/%Y')
plt.gca().xaxis.set_major_formatter(date_format)

plt.legend()


plt.tight_layout() 

plt.show()

#############################################################################
##################### Dayli and accumulate return ###########################
#############################################################################


r1=s1.pct_change()       

ra= ((s1/s1.iloc[0])-1)    

print("The accumulate return is "+str(round(ra[-1]*100,2))+"%")

#############################################################################
############################# Volatility ####################################
#############################################################################


std=r1.std()                
std_annual= std*sqrt(252)   
print("\nThe dayli volatilityis " + str(round(std*100,2)) + "%")
print("The annual volatilityis " + str(round(std_annual*100,2)) + "%\n")


vms = s1.rolling(window=S).std()*sqrt(252)
vmm= s1.rolling(window=M).std()*sqrt(252)
vml= s1.rolling(window=L).std()*sqrt(252)

Vol_window=pd.DataFrame()
Vol_window['Short ' + str(S)+ ' days']=vms
Vol_window['Medium ' + str(M)+ ' days']=vmm
Vol_window['Long ' + str(L)+ ' days']=vml

Vol_window.plot()
plt.title(f'Movel Volatility of {stock}')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.gcf().autofmt_xdate()
date_format=mpl_date.DateFormatter('%d/%m/%Y')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
# plt.savefig('1')
plt.show()

#############################################################################
################################# Sharpe ####################################
#############################################################################

A_R = ((1+ra[-1])**(252/len(ra)))-1   
Selic=0.1365
Sharpe= (A_R-Selic)/std_annual
print('The Sharpe of the stock is: ' + str(Sharpe) + '\n')

#############################################################################
################################ Correlation ################################
#############################################################################

C = [str(stock),'^BVSP',"IEF",'^GSPC','BRL=X','EURBRL=X']
Wallet = pd.DataFrame()
for Stock in C:
    Wallet[Stock]=yf.download(Stock,start=start)['Adj Close'].ffill()
cm=Wallet.iloc[:,:2].pct_change().rolling(window=S).corr()
cr=Wallet.pct_change().corr()

sns.heatmap(cr ,  annot=True)
plt.show()

cm=pd.DataFrame(cm)
t=0.001
cm=cm.mask(np.abs(cm-1)<t)
cm=cm[stock].dropna()
cm=pd.DataFrame(cm)
x=cm.copy()

x.reset_index(inplace=True)
x['Date']=pd.to_datetime(x['Date'])
x=x['Date']
plt.plot_date(x,cm,label=(str(C[0])+' & '+str(C[1])+' '+str(S)+' days window'),linestyle='solid',color='k',marker='')
plt.xlabel('Date')
plt.ylabel('Correlation')
plt.title('Correlation Graph')
plt.legend()
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.show()

#############################################################################
################################ Beta #######################################
#############################################################################

Wallet_r=Wallet.pct_change().dropna()
Beta=(Wallet_r[str(C[1])].cov(Wallet_r[str(C[0])]))/(Wallet_r[str(C[0])].var()) 
print('\nThe beta of '+str(C[1])+' is ' + str(round(Beta,2))+'\n')

#############################################################################
################################# VAR #######################################
#############################################################################


VaR= 0.01

print('The VaR in '+ str(VaR*100)+'% of the bench is: ' + str(np.quantile(Wallet_r[str(C[0])], VaR))) 
print('The VaR in '+ str(VaR*100)+'% of the stock1 is: ' + str(np.quantile(Wallet_r[str(C[1])], VaR))+"\n")

print('The VaR in 1%, 5%, 10% of the '+str(C[0])+' is subsequently: ' + str(np.quantile(Wallet_r[str(C[0])], [0.01,0.05,0.1]))) 
print('The VaR in 1%, 5%, 10% of the '+str(C[1])+' is subsequently: ' + str(np.quantile(Wallet_r[str(C[1])], [0.01,0.05,0.1]))+"\n")

bv=list(np.quantile(Wallet_r[str(C[0])], [0.01,0.05,0.1]))
bs= list(np.quantile(Wallet_r[str(C[1])], [0.01,0.05,0.1]))
df = pd.DataFrame(
{
    str(C[0]): bv ,
    str(C[1]):bs,
}

)

df.plot.bar()
plt.title(str(C[0])+'  &  '+str(C[1]) +' VaR')
plt.xlabel('VaR 1%-10%')
plt.ylabel('Maximum lost')
plt.gcf().autofmt_xdate()
plt.tight_layout()

plt.show()


var2=pd.DataFrame()
for Stock in C:
    var2[Stock]= list(np.quantile(((yf.download(Stock,start=start,end=end)['Adj Close']).pct_change()).dropna(),[0.01,0.05,0.1]))
var2.plot.bar()
plt.title('Wallet VaR')
plt.xlabel('VaR 1%-10%')
plt.ylabel('Maximum lost')
plt.xticks([0,1,2])
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.show()


##############################################################################
########################## Maximum draw down #################################
##############################################################################

ra=(s1/s1.iloc[0])-1
MDD=(s1-s1.cummax())/s1.cummax()
print(f'{MDD.min():.2f}%')
MDD.plot(color='k')
plt.title(f'Maximum Draw Down {stock}')
plt.ylabel('Maximum lost')
plt.xlabel('Date')
plt.tight_layout()
plt.gcf().autofmt_xdate()
date_format=mpl_date.DateFormatter('%d/%m/%Y')
plt.gca().xaxis.set_major_formatter(date_format)
plt.show()


#############################################################################
########################## Normalize Return #################################
#############################################################################

Normalize_return=pd.DataFrame()

for collum in Wallet:                                                    
    Normalize_return[collum] =(Wallet[collum].ffill()/Wallet[collum].ffill().iloc[0])*100  

Normalize_return.plot()
plt.title("Normalize Return")
plt.ylabel('Price Normalized')
plt.xlabel('Date')
plt.gcf().autofmt_xdate()
date_format=mpl_date.DateFormatter('%d/%m/%Y')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.show()  

rnb = (Wallet[str(C[0])]/Wallet[str(C[0])][0])*100 
rns = (Wallet[str(C[1])]/Wallet[str(C[1])][0])*100
RN = pd.DataFrame()
RN[str(C[1])]=(rns) 
RN[str(C[0])] =(rnb)  
RN.plot()
plt.title("Normalize Return")
plt.ylabel('Price Normalized')
plt.xlabel('Date')
plt.gcf().autofmt_xdate()
date_format=mpl_date.DateFormatter('%d/%m/%Y')
plt.gca().xaxis.set_major_formatter(date_format)

plt.tight_layout()
plt.show()  
sys.exit()