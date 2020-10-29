import itertools
import pickle

import bagpipes as pipes
import matplotlib as mpl
import matplotlib.pyplot as plt
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

# making a grid of models with different redshifts
def exp_galaxy_grid(avs=np.array([0.0]), ages=np.array([0.0]), taus=np.array([0.0]), redshifts=np.array([0.0])):
    model_dict = {}
    model_dict['properties'] = []
    model_dict['model'] = []

    for properties in itertools.product(avs, ages, taus, redshifts):
        print('(av,age,tau,redshift)', properties)

        exp = {}
        exp["age"] = properties[1]
        exp["tau"] = properties[2]
        exp["massformed"] = 10.0
        exp["metallicity"] = 0.02

        dust = {}
        dust["type"] = "Calzetti"
        dust["Av"] = properties[0]
        dust["eta"] = 2.

        nebular = {}
        nebular["logU"] = -3.

        model_components = {}
        model_components["redshift"] = properties[3]
        model_components["exponential"] = exp
        model_components["dust"] = dust
        model_components["nebular"] = nebular
        model_components["veldisp"] = 100.0

        model = pipes.model_galaxy(model_components, filt_list=filter_list_splus)
        # print(model.spectrum_full[0])
        # print(model.redshifted_wavs)
        # fig = model.plot()
        model_dict['properties'].append(properties)
        model_dict['model'].append(model)
        # print(model_dict['properties'])

    pickle.dump(model_dict, file=open('exp_model_grid.h5', 'wb'))

def galaxy_fit(galaxy_model, galaxy_properties, fit_instructions,i):
    def load_data(id):
        if id == "galaxy" + str(i):
            return np.array([galaxy_model.photometry, 0.1 * galaxy_model.photometry]).transpose()

    # making galaxy objects with models to create synthetic galaxies
    galaxy_to_fit = pipes.galaxy("galaxy" + str(i), load_data, filt_list=filter_list_splus,
                                 phot_units=galaxy_model.phot_units, spectrum_exists=False)
    fit = pipes.fit(galaxy_to_fit, fit_instructions, run="exp_sfh")

    fit.fit(verbose=False)
    print('(av,age,tau,redshift)', galaxy_properties)
    fig = fit.plot_spectrum_posterior(save=True, show=False)
    fig = fit.plot_sfh_posterior(save=True, show=False)
    fig = fit.plot_corner(save=True, show=False)

    fig = plt.figure(figsize=(12, 7))
    gs = mpl.gridspec.GridSpec(7, 4, hspace=3., wspace=0.1)

    ax1 = plt.subplot(gs[:4, :])

    pipes.plotting.add_observed_photometry(fit.galaxy, ax1, zorder=10)
    pipes.plotting.add_photometry_posterior(fit, ax1)

    labels = ["sfr", "exponential:massformed", "dust:Av", "exponential:metallicity"]

    post_quantities = dict(zip(labels, [fit.posterior.samples[l] for l in labels]))

    axes = []
    for i in range(4):
        axes.append(plt.subplot(gs[4:, i]))
        pipes.plotting.hist1d(post_quantities[labels[i]], axes[-1], smooth=True, label=labels[i])
    plt.savefig('pipes/plots/exp_sfh/galaxy' + str(i) + '_mix.png')

def galaxy_broad_fit(galaxy_model, galaxy_properties, fit_instructions,i):
    def load_data_broad(id):
        if id == "galaxy_broad" + str(i):
            return np.array([galaxy_model.photometry, 0.1 * galaxy_model.photometry]).transpose()[-5:]

    galaxy_to_fit_broad = pipes.galaxy("galaxy_broad" + str(i), load_data_broad, filt_list=filter_list_splus_broad,
                                       phot_units=galaxy_model.phot_units, spectrum_exists=False)

    fit = pipes.fit(galaxy_to_fit_broad, fit_instructions, run="exp_sfh_broad")
    print('(av,age,tau,redshift)', galaxy_properties)
    fit.fit(verbose=False)

    fig = fit.plot_spectrum_posterior(save=True, show=False)
    fig = fit.plot_sfh_posterior(save=True, show=False)
    fig = fit.plot_corner(save=True, show=False)

    fig = plt.figure(figsize=(12, 7))
    gs = mpl.gridspec.GridSpec(7, 4, hspace=3., wspace=0.1)

    ax1 = plt.subplot(gs[:4, :])

    pipes.plotting.add_observed_photometry(fit.galaxy, ax1, zorder=10)
    pipes.plotting.add_photometry_posterior(fit, ax1)

    labels = ["sfr", "exponential:massformed", "dust:Av", "exponential:metallicity"]

        post_quantities = dict(zip(labels, [fit.posterior.samples[l] for l in labels]))

        axes = []
        for i in range(4):
            axes.append(plt.subplot(gs[4:, i]))
            pipes.plotting.hist1d(post_quantities[labels[i]], axes[-1], smooth=True, label=labels[i])
    plt.savefig('pipes/plots/exp_sfh_broad/galaxy' + str(i) + '_broad_mix.png')

def galaxy_sint_fit(galaxy_model, galaxy_properties, fit_instructions,i):
    def load_data_sint(id):
        if id == "galaxy_sint" + str(i):
            for j in range(0, len(filter_list_splus)):
                galaxy_model.photometry[j] = np.random.normal(galaxy_model.photometry[j],
                                                              2 * 0.1 * galaxy_model.photometry[j], 1)
                return np.array([galaxy_model.photometry, 0.1 * galaxy_model.photometry]).transpose()

    galaxy_to_fit_sint = pipes.galaxy("galaxy_sint" + str(i), load_data_sint, filt_list=filter_list_splus,
                                      phot_units=galaxy_model.phot_units, spectrum_exists=False)
    fit = pipes.fit(galaxy_to_fit_sint, fit_instructions, run="exp_sfh_sint")
    print('(av,age,tau,redshift)', galaxy_properties)
    fit.fit(verbose=False)
    fig = fit.plot_spectrum_posterior(save=True, show=False)
    fig = fit.plot_sfh_posterior(save=True, show=False)
    fig = fit.plot_corner(save=True, show=False)

    fig = plt.figure(figsize=(12, 7))
    gs = mpl.gridspec.GridSpec(7, 4, hspace=3., wspace=0.1)

    ax1 = plt.subplot(gs[:4, :])

    pipes.plotting.add_observed_photometry(fit.galaxy, ax1, zorder=10)
    pipes.plotting.add_photometry_posterior(fit, ax1)

    labels = ["sfr", "exponential:massformed", "dust:Av", "exponential:metallicity"]

    post_quantities = dict(zip(labels, [fit.posterior.samples[l] for l in labels]))

    axes = []
    for i in range(4):
        axes.append(plt.subplot(gs[4:, i]))
        pipes.plotting.hist1d(post_quantities[labels[i]], axes[-1], smooth=True, label=labels[i])
    plt.savefig('pipes/plots/exp_sfh_sint/galaxy' + str(i) + '_sint_mix.png')


#####################################################################################################################
# Creating models that will be used to characterization and synthetic fitting

ages = np.array([0.01, 0.1, 1., 10])
taus = np.array([4.])
avs = np.array([0.])  # , 0.2, 0.4, 0.6, 0.8, 1.])
redshifts = np.array([0.02, 0.05])  # , 0.002, 0.5])

# exp_galaxy_grid(avs,ages,taus,redshifts)
#####################################################################################################################

# defining fit parameteres
exp = {}
exp["age"] = (0.001, 15.)
exp["tau"] = (0.3, 10.)
exp["massformed"] = (1., 15.)  # 10
exp["metallicity"] = (0., 2.5)  # 0.02

dust = {}
dust["type"] = "Calzetti"
dust["Av"] = (0., 3) #0
dust["eta"] = 2

nebular = {}
nebular["logU"] = -3.

fit_instructions = {}
fit_instructions["redshift"] = 0.03  # (0.,10.)
fit_instructions["exponential"] = exp
fit_instructions["dust"] = dust
fit_instructions["nebular"] = nebular
#####################################################################################################################
galaxy_list = pickle.load(file=open('exp_model_grid.h5', 'rb'))  # loading galaxy models

# for i in range(0,len(galaxy_list['properties'])):
i = 2
galaxy_fit(galaxy_list['model'][i], galaxy_list['properties'][i], fit_instructions, i)

# galaxy_broad_fit(galaxy_list['model'][i], galaxy_list['properties'][i], fit_instructions,i)
# galaxy_sint_fit(galaxy_list['model'][i], galaxy_list['properties'][i], fit_instructions,i)

# rodar 10 vezes pra uma galáxia normal e com ruído
#mudar rúidos e rodar novamente
