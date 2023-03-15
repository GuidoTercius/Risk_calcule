#############################################################################
################################# Importing #################################
#############################################################################

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from  math import *
from datetime import datetime,timedelta 
from matplotlib import dates as mpl_date
import sys
# sys.path.insert(0,'/Users/GuidoImpactus/Documents/Python/Python_arquivos_Liga/Learning_PythonGirraf/')
# import MG2
plt.style.use('seaborn') # Chossing the style that i will plot the graphics, print(plt.style.available) to know the avaible ones
#Styles that i like 'classic' , 'fast', 'fivethirtyeight' (is kind of fat), 'ggplot', 'seaborn-v0_8-deep' and 'tableau-colorblind10'

#############################################################################
############################## Metrics of risk ##############################
#############################################################################

start = "1900-02-06"
end= "2023-03-13"         ###If you don't put a end he will considerate today!
stock = "BEEF3.SA"
s1=yf.download(stock , start=start )['Adj Close'].ffill()   ### you download the hist of the stock ['Adj Close'] we filter only ['x'] Collumn


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

# plt.plot(SMm['Average Moving '+ str(L) + ' days'], color='#161925')
# plt.plot(SMm['Average Moving '+ str(M) + ' days'], color='#628395')
# plt.plot(SMm['Average Moving '+ str(S) + ' days'],linestyle='--', color='#CBF7ED')
# plt.plot(SMm['Price'], color='r')
SMm.plot()

plt.title('Price Change of: ' + stock)
plt.xlabel('Date')
plt.ylabel('Price')
# plt.xticks(rotation=90)
plt.gcf().autofmt_xdate()
date_format=mpl_date.DateFormatter('%d/%m/%Y')
plt.gca().xaxis.set_major_formatter(date_format)

plt.legend()
# plt.grid(False)

plt.tight_layout() ## get the plot in good demintions
# plt.savefig('a')
plt.show()

#############################################################################
##################### Dayli and accumulate return ###########################
#############################################################################


r1=s1.pct_change()          ### pct_change()calculate the dayli change of the  stock

ra= ((s1/s1.iloc[0])-1)     ### tanking (the first price and divinding all the price)-1 we get the acummulate return
# print(ra[-1])
print("The accumulate return is "+str(round(ra[-1]*100,2))+"%")

#############################################################################
############################# Volatility ################################
#############################################################################


std=r1.std()                ### .std() calculate the standard deviation of the daily returns
std_annual= std*sqrt(252)   ### multiplaing by sqrt(252) we get the annual std!
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

A_R = ((1+ra[-1])**(252/len(ra)))-1  ### [-1]take the last iten from the list **to 252/nummber of days we have the cottation 
Selic=0.1365
Sharpe= (A_R-Selic)/std_annual
print('The Sharpe of the stock is: ' + str(Sharpe) + '\n')
# Sharpe = ra[-1]              ### Sharpe = (annual Return-Selic)/anual volatility
                               ### Annua_return =  (1 + Return acummulate)**(len(ra)/252)

#############################################################################
################################ Correlation ################################
#############################################################################

C = [str(stock),'^BVSP',"IEF",'^GSPC','BRL=X','EURBRL=X']
Wallet = pd.DataFrame()
for Stock in C:
    Wallet[Stock]=yf.download(Stock,start=start)['Adj Close'].ffill()
cm=Wallet.iloc[:,:2].pct_change().rolling(window=S).corr()
cr=Wallet.pct_change().corr()
# sns.heatmap(cr ,  annot=True)

### Take in out the 1 of the correlation!
cm=pd.DataFrame(cm)
t=0.001
cm=cm.mask(np.abs(cm-1)<t)
cm=cm[stock].dropna()

### Getting a x axis for date
cm=pd.DataFrame(cm)
x=cm.copy()

x.reset_index(inplace=True)
x['Date']=pd.to_datetime(x['Date'])
x=x['Date']


plt.plot_date(x,cm,label=(str(C[0])+' & '+str(C[1])+' '+str(S)+' days window'),linestyle='solid',color='k',marker='')

plt.xlabel('Date')
plt.ylabel('Correlation')
plt.title('Correlation Graph')
# plt.xticks(rotation=90)
plt.legend()
plt.gcf().autofmt_xdate() ## it format the date in x axis!
# date_format=mpl_date.DateFormatter('%Y/%m/%d')
# plt.gca().xaxis.set_major_formatter(date_format)

plt.tight_layout()
# plt.savefig("2")

plt.show()

#############################################################################
################################ Beta #######################################
#############################################################################


Wallet_r=Wallet.pct_change().dropna()

Beta=(Wallet_r[str(C[1])].cov(Wallet_r[str(C[0])]))/(Wallet_r[str(C[0])].var()) ### covariation betwen Sotck and Bench / by variation of the Bench
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
# print(df)
df.plot.bar()
plt.title(str(C[0])+'  &  '+str(C[1]) +' VaR')
plt.xlabel('VaR 1%-10%')
plt.ylabel('Maximum lost')
plt.gcf().autofmt_xdate()
plt.tight_layout()
# plt.savefig('3')
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
# plt.savefig('4')
plt.show()
# get the quantile (element , what quantile)0.01 = 1% chance of return in a day

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
# plt.savefig('5')
plt.show()
# MDD2=pd.DataFrame()
# for Stock in C:
#     MDD2[Stock]= ((((yf.download(Stock,start=start)['Adj Close']).dropna())-((yf.download(Stock,start=start)['Adj Close'].dropna()).max()))/((yf.download(Stock,start=start)['Adj Close']).dropna()).max()).min()
# print(MDD2)
# for Stock in C:
#     AC=yf.download(Stock, start=start)
#     maximum_drawdown= (AC - AC.max())/AC.max()
#     MDD2[Stock]= maximum_drawdown
# print(MDD2)
# MDD2.plot()
# plt.title('Maximum Draw Down')
# plt.ylabel('Maximum lost')
# plt.xlabel('Date')
# plt.tight_layout()
# plt.show()


#############################################################################
########################## Normalize Return #################################
#############################################################################

Normalize_return=pd.DataFrame()

for collum in Wallet:                                                      ### utilazing a Looping to passe the diferent stocks to a DataFrame
    Normalize_return[collum] =(Wallet[collum].ffill()/Wallet[collum].ffill().iloc[0])*100  ### Dividing the collum by her first item!
    
Normalize_return.plot()
plt.title("Normalize Return")
plt.ylabel('Price Normalized')
plt.xlabel('Date')
plt.gcf().autofmt_xdate()
date_format=mpl_date.DateFormatter('%d/%m/%Y')
plt.gca().xaxis.set_major_formatter(date_format)

plt.tight_layout()
# plt.savefig('6')
plt.show()  



rnb = (Wallet[str(C[0])]/Wallet[str(C[0])][0])*100 #Dividing be it self and multiplying by 100 you get the normalize return
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
# plt.savefig('7')
plt.show()  
sys.exit()