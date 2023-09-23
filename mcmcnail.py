# R. Walker, created April 29 2023, last updated Aug 15 2023
# inputs: KMTNet 2021 event number to model, desired MCMC number of walkers, steps to take, and steps to burn
# opt variable sets: sub_jd, sub_hjd
# outputs: event data file, MCMC distribution chain, corner.py plot of predictive model results
# manipulates folders "mcmcfigs", "mcmchains", "mcmcorners", "eventfiles"

import os
import subprocess
import wget
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import emcee
import corner

def nail(num):
    ########################### user variable setup (changed  most often)
    num = num
    nwalkers = 100
    steps = 500
    burn = 50
    sub_jd = 2450000. # subtracted from prediction x data, turns hjd into jd
    
    print("Event "+str(num)+": ")

    ########################### file setup
    researchfolder = '/your/path/where/you/save/this/project' # edit me!
    
    padnum = "%04d" % num
    urlbeg = "https://kmtnet.kasi.re.kr/~ulens/event/2021/" # base webpage that holds all basic parameters
    eventdata = pd.read_html(urlbeg)[0]
    fieldno = str(eventdata[1][num])
    url = urlbeg+"data/KB21"+str(padnum)+"/pysis/KMTC"+fieldno[3:5]+"_I.pysis"
    path = "KMTC"+fieldno[3:5]+"_I.pysis"
    if os.path.isfile(researchfolder+'/eventfiles/'+path)==False:
        subprocess.Popen(["wget",url], cwd=researchfolder+'/eventfiles')
    
    ########################### variable setup
    path = "KMTC"+fieldno[3:5]+"_I.pysis"
    df = pd.DataFrame(data=pd.read_csv(researchfolder+'/eventfiles/'+path,sep='\s+', skiprows=1, names=['HJD','Delta_flux','flux_err','mag','mag_err','fwhm','sky','secz']))
    # have to skip one row and rename files because jupyter reads  .pysis files differently
    x = df.HJD
    y = df.mag
    yerr = df.mag_err
    jddates = pd.read_csv("alertdates.csv")
    df_forpred = df[df.HJD<jddates.JD[num]-sub_jd]
    sub_hjd = float(df_forpred.HJD.iloc[-1]) # subtracted from t0 and jd, makes numbers more manageable 
    
    # archival data numbers uses to confirm prediction(s)
    t0 = float(eventdata[7][num])-sub_hjd
    tE = float(eventdata[8][num])
    u0 = float(eventdata[9][num])
    ms = float(eventdata[10][num])
    m0 = float(eventdata[11][num])
    lininit = [t0,tE,u0,m0,fs(m0,ms)] # initial linear parameter guesses
    loginit = [t0,np.log10(tE),np.log10(u0),m0,np.log10(fs(m0,ms))] # initial log parameter guesses
    
    # fitting model to archival data using curvefit, best-fit parameters stored in 'popt'
    popt,pcov = opt.curve_fit(logm,x-sub_hjd,y,p0=loginit,sigma=df.mag_err, bounds=([-100., 1., -4., 15., -3.], [200., 2., .5, 25., .5])) # order: t0, log(tE), log(u0), m0, log(fs)
    
    ########################### prediction setup
    pred_t0 = [df_forpred.HJD[0]-sub_hjd+i for i in [10,20,30,40,50,60]] # should be just ahead of data start
    pred_tE = [10,30,50,70,90,110]
    pred_u0 = 0.1
    pred_fs = 0.9
    
    ########################### plot archival against prediction 
    figtree = plt.figure()
    plt.xlabel('Time (HJD)')
    plt.xlim(min(x),max(x))
    plt.ylabel('Magnitude')
    plt.ylim([max(y)+2.,min(y)-2.]) #it just looks better with the extra space to me
    plt.title('Model Comparison')
    plt.scatter(x, y, marker='.', label='Full Data', s=4)
    plt.scatter(df_forpred.HJD, df_forpred.mag, marker='.', label='Pred Data', s=4)
    plt.plot(x,logm(x-sub_hjd,*popt), label='Full Model',color='b', lw=3) 
    plt.legend()
    for j in np.arange(0,6,1):
        var = j+1
        pred_loginit = [pred_t0[j],np.log10(pred_tE[j]),np.log10(pred_u0),m0,np.log10(pred_fs)]
        #print(pred_loginit) #for if the bounds get mad at you
        pred_popt,pred_pcov = opt.curve_fit(logm,df_forpred.HJD-sub_hjd,df_forpred.mag,p0=pred_loginit,sigma=df_forpred.mag_err, bounds=([-150., 1., -4., 12., -3.], [150., 2.5, 1., 25., 0.])) # order: t0, log(tE), log(u0), m0, log(fs)
        plt.plot(x,logm(x-sub_hjd,*pred_popt), alpha=.7, linewidth=var*.7)
        # the prediction functions are wonky, must fix, might be due to subtraction from hjd dates for df_forpred
    figtree.savefig("mcmcfigs/fig_"+str(num)+".pdf")
    plt.close(figtree)
    
    ########################### MCMC distribution 
    ndim = len(pred_popt)
    pos = [pred_popt + 1.0e-2 * np.random.randn(ndim) for i in range(nwalkers)]
    
    filename = "mcmchains/chain_"+str(num)+".h5"
    backend = emcee.backends.HDFBackend(filename) # setting up chain save that's about to be generated
    backend.reset(nwalkers, ndim)
    
    sampler = emcee.EnsembleSampler(nwalkers,ndim,log_probability,args=(df_forpred.HJD-sub_hjd,df_forpred.mag,df_forpred.mag_err),backend=backend)
    sampler.run_mcmc(pos,steps,progress=True);
    samples = sampler.get_chain(discard=burn, thin=20, flat=True).reshape((-1,ndim))
    
    # mcmc chain production
    fig, axes = plt.subplots(len(pred_popt), figsize=(10, 7), sharex=True)
    samples_plot = sampler.get_chain()
    labels = ["t0","log(tE)","log(u0)","m0","log(fs)"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples_plot[:, :, i], alpha=0.3)
        ax.set_xlim(0, len(samples_plot))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("Step Number");
    fig.savefig("mcmchains/chain_"+str(num)+".pdf")
    plt.close(fig)
    
    # corner plot of mcmc
    figure = corner.corner(
        samples,
        labels=["t0","log(tE)","log(u0)","m0","log(fs)"],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
    );
    figure.savefig("mcmcorners/corner_"+str(num)+".pdf")
    plt.close(figure)
    
    timescale = x.iloc[-1]-x.iloc[0]
    return(print("Timescale of event = ",timescale,"\n")) # in days

##################################################### functions used for calculations

def a(t,t0,tE,u0): # returns total magnification A
    tau = (t-t0)/tE
    u = np.sqrt(tau**2 + u0**2)
    return ((u**2+2)/(u*np.sqrt(u**2+4)))

def m(t,t0,tE,u0,m0,fs): # returns magnitude
    amp = a(t,t0,tE,u0)
    return (m0 - 2.5*np.log10(1+fs*(amp-1)))

def fs(m0,ms):
    return (pow(10,((m0-ms)/2.5)))

##################################################### modeling functions

def m(t,t0,tE,u0,m0,fs): #uses /LINEAR/ parameters
    return (m0 - 2.5*np.log10(1+fs*(a(t,t0,tE,u0)-1)))

def logm(t,t0,log_tE,log_u0,m0,log_fs): #uses /LOGARITHMIC/ parameters
    if log_tE > 10.:
        return np.zeros(shape=t.shape)
    else: tE = 10**(log_tE)
    if log_u0 > 2.:
        return np.zeros(shape=t.shape)
    else: u0 = 10**(log_u0)
    if log_fs > 1.:
        return np.zeros(shape=t.shape)
    else: fs = 10**(log_fs)
    return m(t,t0,tE,u0,m0,fs)

##################################################### EMCEE setup functions

def log_prior(theta): 
    t0,log_tE,log_u0,m0,log_fs = theta
    if t0 < -200. or t0 > 200.:
        return -1.0e18
    if log_tE < 0. or log_tE > 3.:
        return -1.0e18
    if log_u0 < -4. or log_u0 > .5:
        return -1.0e17
    if m0 < 15. or m0 > 22.:
        return -1.0e18
    if log_fs < -3. or log_fs > .5:
        return -1.0e18
    return 0.0

# the bad and ret in this function were for testing purposes, too scared to remove it though
def log_likelihood(theta, x, y, yerr):
    bad = 0
    t0,log_tE,log_u0,m0,log_fs = theta
    if t0 < -200. or t0 > 200.:
        bad = 1
        ret = -1.0e18
    if log_tE > 3. or log_tE < -1.:
        bad = 1
        ret = -1.0e18
    else: 
        tE = 10**log_tE
    if log_u0 < -3. or log_u0 > .5: 
        bad = 1
        ret = -1.0e18
    else: 
        u0 = 10**log_u0
    if m0 < 15. or m0 > 22.:
        bad = 1
        ret = -1.0e18
    if log_fs > .5 or log_fs < -3.:
        bad = 1
        ret = -1.0e18
    else: 
        fs = 10**log_fs
    if bad == 1:
        return ret
    model = m0 - 2.5*np.log10(1+fs*(a(x,t0,tE,u0)-1))
    sigma_squared = yerr**2
    ret = -.5 * np.sum((y-model)**2/sigma_squared)
    return ret

def log_probability(theta, x, y, yerr):
    if not np.isfinite(log_prior(theta)): 
        return -np.inf
    return log_prior(theta) + log_likelihood(theta, x, y, yerr)