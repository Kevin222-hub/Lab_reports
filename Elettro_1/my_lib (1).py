import matplotlib.pyplot as plt
# per manipolare arrays e fare di conto
import numpy as np
from math import log10, floor, sqrt
# per fare cose piu' sofisticate come per esempio generare 
#numeri pseudo random
import scipy as sp
from scipy import stats


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

# funzione che calcola m, c, sd_m, sd_c, cov_mc a partire da
# x, y, sd_y
def lin_fit(x, y, sd_y, verbose=True, plot=False):
    m = my_cov(x, y, sd_y) / my_var(x, sd_y)
    var_m = 1 / ( my_var(x, sd_y) * np.sum( np.power(sd_y, -2)) )
    c = np.mean(y) - np.mean(x) * m
    var_c = my_mean(x*x, sd_y)  / ( my_var(x, sd_y) * np.sum( np.power(sd_y, -2)))
    cov_mc = - my_mean(x, sd_y) / ( my_var(x, sd_y) * np.sum( np.power(sd_y, -2))) 
    if (verbose):
        print ('m         = ', m)
        print ('sigma(m)  = ', np.sqrt(var_m))
        print ('c         = ', c)
        print ('sigma(c)  = ', np.sqrt(var_c))
        print ('cov(m, c) = ', cov_mc)
        
    if (plot):
        # rappresento i punti misurati
        plt.errorbar(x, y, yerr=sd_y, xerr=0, ls='', marker='.', 
                     label='punti misurati')

        # costruisco dei punti x su cui valutare la retta di regressione
        xmin = np.min(x)
        xmax = np.max(x)
        xl = np.linspace(0.8*xmin-.2*(xmax-xmin), xmax*1.2+.2*(xmax-xmin), 100)
        # uso i parametri medi di m e c
        yl = my_line(xl, m, c)
        # rappresento la retta di regressione
        plt.plot(xl, yl, 'g-.', label='retta di regressione')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('Regressione lineare')
        a=plt.legend()
    return m, np.sqrt(var_m), c, np.sqrt(var_c), cov_mc
    
#Grafico Fit
def fit_graph(x,y,sy,m,sm,c,sc,co, grid=True, err=True):
    xmin = np.amin(x)
    xmax = np.amax(x)
    # rappresento i punti misurati
    plt.errorbar(x, y, yerr=sy, xerr=0, ls='', marker='.', label='punti misurati')

    # costruisco dei punti x su cui valutare la retta di regressione
    xl = np.linspace(xmin, xmax, 100)
    # uso i parametri medi di m e c
    yl = my_line(xl, m,c)
    # rappresento la retta di regressione
    plt.plot(xl, yl, 'g-.', label='retta di regressione')# propagazione incertezza su y a partire da m e c
    # incertezza sulle y
    yinc = y_inc(xl, sm, sc, co)
    if (err): 
        # curve a y +- incertezza
        plt.plot(xl, yl+yinc, 'r--', label='banda incertezza $\pm \sigma_{m}$')
        plt.plot(xl, yl-yinc, 'r--')   
    if (grid):
        plt.grid(b=None, which='major', axis='both')
    plt.legend()