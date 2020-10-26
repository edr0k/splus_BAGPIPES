import bagpipes as pipes
import numpy as np

filter_list_splus = ['filters/F0378.dat',
                     'filters/F0395.dat',
                     'filters/F0410.dat',
                     'filters/F0430.dat',
                     'filters/F0515.dat',
                     'filters/F0660.dat',
                     'filters/F0861.dat',
                     'filters/uJAVA.dat',
                     'filters/gSDSS.dat',
                     'filters/rSDSS.dat',
                     'filters/iSDSS.dat',
                     'filters/zSDSS.dat']

filter_list_splus_broad = ['filters/uJAVA.dat',
                           'filters/gSDSS.dat',
                           'filters/rSDSS.dat',
                           'filters/iSDSS.dat',
                           'filters/zSDSS.dat']


exp = {}                          # Tau model star formation history component
exp["age"] = 6.                   # Gyr
exp["tau"] = 4.                   # Gyr
exp["massformed"] = 9.            # log_10(M*/M_solar)
exp["metallicity"] = 0.5          # Z/Z_oldsolar

dust = {}                         # Dust component
dust["type"] = "Calzetti"         # Define the shape of the attenuation curve
dust["Av"] = 0.2                  # magnitudes
dust["eta"] = 2.                  # Extra dust for young stars: multiplies Av

nebular = {}                      # Nebular emission component
nebular["logU"] = -3.             # log_10(ionization parameter)

model_components = {}                   # The model components dictionary
model_components["redshift"] = 0.015     # Observed redshift
model_components["exponential"] = exp
model_components["dust"] = dust
model_components["t_bc"] = 0.01         # Lifetime of birth clouds (Gyr)
model_components["veldisp"] = 200.      # km/s
model_components["nebular"] = nebular

model_z0 = pipes.model_galaxy(model_components, filt_list=filter_list_splus)

fig = model_z0.plot()
fig = model_z0.sfh.plot()


def load_data(id):
    if id == 'z0':
        return np.array([model_z0.photometry, 0.1 * model_z0.photometry]).transpose()

def load_data_broad(id):
    if id == 'z0_broad':
        return np.array([model_z0.photometry, 0.1 * model_z0.photometry]).transpose()[-5:]

galaxy_to_fit = pipes.galaxy('z0', load_data, filt_list=filter_list_splus, phot_units=model_z0.phot_units,
                             spectrum_exists=False)

galaxy_to_fit_broad = pipes.galaxy('z0_broad', load_data_broad, filt_list=filter_list_splus_broad, phot_units=model_z0.phot_units,
                                   spectrum_exists=False)

exp = {}
exp["age"] = (0.1, 15.)
exp["tau"] = (0.3, 10.)                   
exp["massformed"] = (1., 15.)             
exp["metallicity"] = (0., 2.5)

dust = {}
dust["type"] = "Calzetti"
dust["Av"] = (0., 3)

nebular = {}
nebular["logU"] = -3.

dust["eta"] = 2.

fit_instructions = {}                     
fit_instructions["redshift"] = 0.015
fit_instructions["exponential"] = exp
fit_instructions["dust"] = dust
fit_instructions["nebular"] = nebular

fit = pipes.fit(galaxy_to_fit, fit_instructions)
fit_broad = pipes.fit(galaxy_to_fit_broad, fit_instructions)

fit.fit(verbose=True)
fit_broad.fit(verbose=True)

fig = fit.plot_spectrum_posterior(save=False, show=True)
fig = fit.plot_sfh_posterior(save=False, show=True)
fig = fit.plot_corner(save=False, show=True)

fig = fit_broad.plot_spectrum_posterior(save=False, show=True)
fig = fit_broad.plot_sfh_posterior(save=False, show=True)
fig = fit_broad.plot_corner(save=False, show=True)



# at z=0.0015

# broad
#Parameter                          Posterior percentiles
                                #16th       50th       84th
#----------------------------------------------------------
#dust:Av                        0.138      0.363      0.709
#exponential:age                1.509      4.471      8.676
#exponential:massformed         8.647      8.915      9.113
#exponential:metallicity        0.282      0.988      1.933
#exponential:tau                3.562      6.589      8.903

# all

#Parameter                          Posterior percentiles
                                #16th       50th       84th
#----------------------------------------------------------
#dust:Av                        0.081      0.218      0.414
#exponential:age                3.919      6.946     10.120
#exponential:massformed         8.851      9.009      9.130
#exponential:metallicity        0.322      0.561      0.958
#exponential:tau                4.249      7.113      9.260

# at z=0.05

# broad bands:

#Parameter                          Posterior percentiles
                                #16th       50th       84th
#----------------------------------------------------------
#dust:Av                        0.122      0.331      0.665
#exponential:age                1.918      4.957      9.025
#exponential:massformed         8.704      8.955      9.125
#exponential:metallicity        0.354      1.032      1.973
#exponential:tau                3.778      6.794      8.904

# broad+narrow bands:

#Parameter                          Posterior percentiles
                                #16th       50th       84th
#----------------------------------------------------------
#dust:Av                        0.097      0.246      0.457
#exponential:age                3.554      6.407      9.769
#exponential:massformed         8.838      9.008      9.128
#exponential:metallicity        0.407      0.767      1.246
#exponential:tau                4.699      7.319      9.126

# input

#exp = {}                          # Tau model star formation history component
#exp["age"] = 6.                   # Gyr
#exp["tau"] = 4.                   # Gyr
#exp["massformed"] = 9.            # log_10(M*/M_solar)
#exp["metallicity"] = 0.5          # Z/Z_oldsolar

#dust = {}                         # Dust component
#dust["type"] = "Calzetti"         # Define the shape of the attenuation curve
#dust["Av"] = 0.2                  # magnitudes
#dust["eta"] = 2.                  # Extra dust for young stars: multiplies Av

