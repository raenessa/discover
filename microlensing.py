import numpy as np

def a(t,t0,tE,u0):
    tau = (t-t0)/tE
    u = np.sqrt(tau**2 + u0**2)
    return ((u**2+2)/(u*np.sqrt(u**2+4)))
# returns total magnification A = (u^2 + 2)/[u*sqrt(u^2 + 4)], u = sqrt((u0^2 + (t-t0)^2)/(tE^2))
# u = angular separation between source and lens, t = time (HJD), u0 = impact parameter of event
# t0 = time of closest alignment when u = u0 (time of max magnification), tE = Einstein ring crossing time
#fs = fraction of light in aperture from the source star

def m(t,t0,tE,u0,m0,fs):
    amp = a(t,t0,tE,u0)
    return (m0 - 2.5*np.log10(1+fs*(amp-1))) # m(t) = m0 - 2.5log(1+fs*(A-1))
# returns magnitude