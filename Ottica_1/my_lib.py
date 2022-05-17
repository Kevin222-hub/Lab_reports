import matplotlib.pyplot as plt
import numpy as np
from math import log10, floor, sqrt, pi
import scipy as sp
import pandas as pd
from scipy import stats,integrate

# Per Derivare:  y = np.gradient(Y,x)
# Per Integrare: Y = integrate.cumtrapz(y, x, initial=0)

#########################################################################
# funzione per arrotondare con un certo numero di cifre significative
#########################################################################
def round_sig(x, sig=2):
        return round(x, sig-int(floor(log10(abs(x))))-1)
########################################################################
def y_inc(xl, sigma_m, sigma_c, cov_mc):
    return np.sqrt(np.power(xl, 2)*np.power(sigma_m, 2) +
                   np.power(sigma_c, 2) +
                  2*xl*cov_mc) 
#########################################################################
# funzione per fare un istogramma con una gaussiana sovrapposta
#########################################################################
#
def gaussHistogram(d, xl='x', yl='y', titolo='titolo', bin_scale=0.5):
    mean = d.mean()
    std = d.std()

# scelta del binning
    binsize = std*bin_scale # metà della standard deviation di default
    interval = d.max() - d.min()
    nbins = int(interval / binsize)
    
# 1) Crea un numpy array con 100 valori equamente separati nell'intervallo voluto dell'asse x
    lnspc = np.linspace(d.min()-std, d.max()+std, 100) 

# in questo modo posso raccogliere in vettori le informazioni sull'istogramma
    counts , bins , patches = plt.hist(d, bins=nbins,
                                       color="blue", alpha=0.75)
    
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(label=titolo)
# ==> Disegna una distribuzione normale

# 2) Normalizza la funzione f(x) in modo che l'integrale da -inf a +inf sia il numero totale di misure
    norm_factor = d.size * binsize

# 3) Crea un numpy array con i valori f(x), uno per ciascun punto
# NOTA: Ho usato la distribuzione normale presa da "scipy" 
#      (vedi all'inizio del programma "from scipy import stats")
    f_gaus = norm_factor*stats.norm.pdf(lnspc, mean, std) 
# draw the function
    plt.plot(lnspc, f_gaus, linewidth=1, color='r',linestyle='--')
    print('counts      = ', len(d))
    print('mean        = ', mean)
    print('sigma       = ', std)
    print('sigma_mean  = ', std/sqrt(len(d)))

#########################################################################
# funzione per fare il fit lineare
#########################################################################
#
# funzione per valore atteso pesato per 1/$\sigma_i^2$
def my_mean(x, w):
    return np.sum( x*np.power(w, -2) ) / np.sum( np.power(w, -2) )

def my_mean_sigma(w):
    return np.sqrt(1/np.sum(np.power(w,-2)))

def my_cov(x, y, w):
    return my_mean(x*y, w) - my_mean(x, w)*my_mean(y, w)

def my_var(x, w):
    return my_cov(x, x, w)

# relazione lineare 
def my_line(x, m=1, c=0):
    return x*m +c

# Chi2 per fit lineare
def my_chi_fit(x,y,sd_y, m,c, plot=False):
    d_f = np.len(x)-2
    chi2 = np.sum(np.power( (y-m*x-c)/sd_y ,2))
    return chi2

# Media Dinamica di una array
def mean_array(Array,dN = 1):
    arr_len = int(len(Array)/dN) #lunghezza arrays
    
    m = np.empty(3, dtype=object)
    m[0] = np.empty(arr_len, dtype=object) # asse x
    m[1] = np.empty(arr_len, dtype=object) # mean y
    m[2] = np.empty(arr_len, dtype=object) # std y
    
   
    for x in range(arr_len):

        N = (x+1)*dN
        m[0][x] = N-1 # asse x il 1° conteggio è 0 
        
        #media
        A = 0
        xi = 0
        while xi < N:  
            A = A + Array[xi] #sommatoria
            xi = xi + 1
        m[1][x] = A/N #calcolo
        
        #std
        A = 0
        xi = 0
        if N == 1:
            m[2][x] = 0
        else:
            while xi < N: 
                A = A + (Array[xi]-m[1][x])**2 #sommatoria
                xi = xi + 1
            m[2][x] = sqrt(A)/(N-1) #calcolo
    return m 



# Somma incertezze No arrays!!!
def sqrt_sum(*arg):
    return np.sqrt(np.sum(np.power(arg,2)))


#Grafico Fit
def my_fit_graph(x,y,sd_y, m,sm,c,sc,cov_mc, grid=True, err=True):
    xmin = np.amin(x)
    xmax = np.amax(x)
    # rappresento i punti misurati
    plt.errorbar(x, y, yerr=sd_y, xerr=0, ls='', marker='.', label='punti misurati')

    # costruisco dei punti x su cui valutare la retta di regressione
    xl = np.linspace(0.8*xmin-.2*(xmax-xmin), xmax*1.2+.2*(xmax-xmin), 100)
    # uso i parametri medi di m e c
    yl = my_line(xl, m,c)
    # rappresento la retta di regressione
    plt.plot(xl, yl, 'g-.', label='retta di regressione')# propagazione incertezza su y a partire da m e c
    # incertezza sulle y
    yinc = y_inc(xl, sm, sc, cov_mc)
    if (err): 
        # curve a y +- incertezza
        plt.plot(xl, yl+yinc, 'r--', label='banda incertezza $\pm \sigma_{m}$')
        plt.plot(xl, yl-yinc, 'r--')   
    if (grid):
        plt.grid(b=None, which='major', axis='both')
    plt.legend()
    
# funzione che calcola m, c, sd_m, sd_c, cov_mc a partire da
# x, y, sd_y
def my_fit( x,y,sd_y, sdx = 0 , verbose=True, plot=False, grid=True, err=True):
   
    m = my_cov(x, y, sd_y) / my_var(x, sd_y)
    var_m = 1 / ( my_var(x, sd_y) * np.sum( np.power(sd_y, -2)) )
    sm = np.sqrt(var_m)
    
    c = np.mean(y) - np.mean(x) * m
    var_c = my_mean(x*x, sd_y)  / ( my_var(x, sd_y) * np.sum( np.power(sd_y, -2)))
    sc = np.sqrt(var_c)
    
    cov_mc = - my_mean(x, sd_y) / ( my_var(x, sd_y) * np.sum( np.power(sd_y, -2))) 

    if isinstance(sdx,(list,pd.core.series.Series,np.ndarray)): # sdx array
        A = np.sqrt( np.power(m*sdx,2) + np.power(sd_y,2) )
        m,sm,c,sc,cov_mc = my_fit(x,y,A, sdx = 0, verbose = verbose , plot = plot, grid = grid, err = err)
    else:
        if sdx != 0:  # sdx costante
            A = np.sqrt( (m*sdx)**2 + np.power(sd_y,2) )
            m,sm,c,sc,cov_mc = my_fit(x,y,A, verbose = verbose , plot = plot, grid = grid, err = err)
        else:
            if (verbose):
                print ('m         = ', m)
                print ('sigma(m)  = ', sm)
                print ('c         = ', c)
                print ('sigma(c)  = ', sc)
                print ('cov(m, c) = ', cov_mc)
            if (plot):
                my_fit_graph(x, y,sd_y, m,sm,c,sc,cov_mc, grid = grid, err=err)
    return m, sm, c, sc, cov_mc
########################################################################################################

# Intersezione con linearizzazione

def SigmaM(x1,sx1,x2,sx2,y1,sy1,y2,sy2):
    
    dy2 =  1/(x2-x1) *sy2
    dy1 = -1/(x2-x1) *sy1
    #dx2 =  ((y2-y1)/(x2-x1)**2)*sx2
    #dx1 =  ((y2-y1)/(x2-x1)**2)*sx1
    dx2 = 0
    dx1 = 0

    sm = sqrt_sum(dx1,dx2,dy1,dy2)
    #print(dy2,dy1,dx2,dx1)
    return sm

def SigmaQ(x1,sx1,x2,sx2,y1,sy1,y2,sy2):
    
    dy2 = -x1/(x2-x1) *sy2
    dy1 =  x2/(x2-x1) *sy1
    #dx2 =  x1*(y2-y1)/np.power(x2-x1,2) *sx2
    #dx1 = -x2*(y2-y1)/np.power(x2-x1,2) *sx1
    dx2 = 0
    dx1 = 0
    sq = sqrt_sum(dx1,dx2,dy1,dy2)
    return sq
def Retta(x1,sx1,x2,sx2,y1,sy1,y2,sy2):
    m = (y2-y1)/(x2-x1)
    q = y1 - m*x1
    sm = SigmaM(x1,sx1,x2,sx2,y1,sy1,y2,sy2)
    sq = SigmaQ(x1,sx1,x2,sx2,y1,sy1,y2,sy2)
    return m,sm,q,sq

def single(Value,x1,x2,y1,y2,sx1,sx2,sy1,sy2):
    m,sm,q,sq = Retta(x1,sx1,x2,sx2,y1,sy1,y2,sy2)
    X = (Value-q)/m
    sX = sqrt_sum( 1/m*sq, ((Value-q)/m**2)*sm) #da rifare
    return X,sX

def check_array(sx_arr,sy_arr,sdx,sdy,a,b):
    if sx_arr:
        sxa = sdx[a]
        sxb = sdx[b]
    else:
        sxa = sdx
        sxb = sdx
    if sy_arr:
        sya = sdy[a]
        syb = sdy[b]
    else:
        sya = sdy
        syb = sdy
    return sxa,sxb,sya,syb

def intersect(Value,x,y,sdx = 0,sdy = 0,plot = False):

    N = np.size(y)
    sx_arr = False
    sy_arr = False
   
    if isinstance(sdx,(list,pd.core.series.Series,np.ndarray)):
        sx_arr = True
    if isinstance(sdy,(list,pd.core.series.Series,np.ndarray)):
        sy_arr = True

    X = np.asarray([])
    sX = np.asarray([])
    
    for i in range(0,N):
        if y[i] == Value:
            X = np.append(X,x[i])
            if sx_arr:
                sX = np.append(sX,sdx[i])
            else:
                sX = np.append(sX,sdx)
       
        if i != N-1:
            if y[i] < Value and y[i+1] > Value:
                sx1,sx2,sy1,sy2 = check_array(sx_arr,sy_arr,sdx,sdy,i,i+1)
                T,S = single(Value,x[i],x[i+1],y[i],y[i+1],sx1,sx2,sy1,sy2)
                X = np.append(X,T)
                sX = np.append(sX,S)

                
            if y[i] > Value and y[i+1] < Value:
                sx1,sx2,sy1,sy2 = check_array(sx_arr,sy_arr,sdx,sdy,i+1,i)
                T,S = single(Value,x[i+1],x[i],y[i+1],y[i],sx1,sx2,sy1,sy2)
                X  = np.append(X,T)
                sX = np.append(sX,S)
    if plot:
        plt.plot(x,y,'.')
        for i in range(0,len(X)):
            plt.vlines(X[i],min(y),max(y),linestyles='dashed',color='g')
    
    return X,sX
##################################################################################################

def chi_graph(chi2, label,DF,acc=0.05, xmax = 0 ):
    
    ACC = stats.chi2.isf(acc, DF) # Valore Accettazione 
    
    if xmax == 0:
        xmax = ACC*1.2
    
    x = np.linspace(0,xmax,1000)
    plt.plot(x, stats.chi2.pdf(x, df=DF), color='b', lw=1, label='Pdf con {} df'.format(DF) )
    plt.vlines(ACC, ymin=0, ymax = stats.chi2.pdf(ACC, df=DF),color = 'r',label = 'Accettazione {}'.format(acc)) #Accettazione
        
    plt.vlines(chi2, ymin=0, ymax = stats.chi2.pdf(chi2, df=DF),label = label) #Valori testati

    plt.grid(b=None, which='major', axis='both')
    plt.xlabel('Value')
    plt.ylabel('Pdf')
    plt.title(r'$\chi^2$ Test')
    plt.legend()
    plt.show()
    return 0