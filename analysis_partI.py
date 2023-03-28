# this should be run in my pymc environment
# in this program, I build a Bayesian model to fit the primer efficiencies for
# all of the data in this directory

# I will plot diagnostics on these estimates, but in practice I may want to
#  combine this inference with additional data to fit my real quantities of interest

import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns

n_chains = 6

infile_qpcr = "long_form_table.csv"
infile_quant = "qPCR_picogram.csv"
with_dil_rands = True
with_samp_primer_rands = False

outprefix="fit_effs"
if with_dil_rands:
    outprefix += "_dil_rands"
if with_samp_primer_rands:
    outprefix += "_primer_rands"

qpcr_tab = pd.read_csv( infile_qpcr )
quant_tab = pd.read_csv( infile_quant )
#loci = pd.read_csv( infile_loci, sep="\t", names=["chr","source","feature","start","end","score","strand","display","attr"])
#loci = loci[loci["feature"] == "qPCR"].copy().reset_index(drop=True)

print("Setting up data for fitting")
# set up some additional useful columns
qpcr_tab["log2dil"] = np.log2( qpcr_tab.relative_concentration )

# now pull out the dilution data that we will use for fitting
# in the process, we discretize the primer IDs since we need those later
dil_dat = qpcr_tab[
        qpcr_tab.type == "dill_ser"
    ].copy(
    ).merge(quant_tab, how="left", on=["sample_name","type","bio_Rep","geno","relative_concentration"])
#print(dil_dat)
dil_dat = dil_dat[~np.isnan(dil_dat["picograms"])].copy()
dil_dat["nanograms"] = dil_dat["picograms"]*1e3
dil_dat["log_mass"] = np.log2(dil_dat["nanograms"])
sns.scatterplot(dil_dat, x="log_mass", y="Cq");
plt.savefig("cq_vs_mass.png")
plt.close()
#print(with_quants)
#print(dil_dat.iloc[0,:])
#sys.exit()

all_primers = list(set(dil_dat["primer_desc"]))
dil_dat["primer_inds"] = [all_primers.index(x) for x in dil_dat["primer_desc"] ]
dil_dat["dil_rand_name"] = dil_dat["date"] + "_" + dil_dat["relative_concentration"].astype("string") + "_" + dil_dat["sample_name"]
dil_rands = list(set(dil_dat["dil_rand_name"]))
dil_dat["dil_rand_ind"] = [dil_rands.index(x) for x in dil_dat["dil_rand_name"]]

# and also pull out the data for ct fitting
ct_dat = qpcr_tab[np.logical_or(qpcr_tab.type == "extract", qpcr_tab.type == "input")].copy()
ct_dat["primer_inds"] = [all_primers.index(x) for x in ct_dat["primer_desc"] ]

## also figure out indexing for other factors that enter into the model
ct_dat["samp_rand_name"] = ct_dat["sample_name"] + ct_dat["date"]
ct_dat["rep_rand_name"] = ct_dat["sample_name"] + ct_dat["bio_Rep"].astype("str")
ct_dat["primer_rand_name"] = ct_dat["primer_desc"] + ct_dat["date"]
ct_dat["primer_geno"] = ct_dat["primer_desc"] + "_" + ct_dat["geno"]
ct_dat["samp_primer"] = ct_dat["sample_name"] + ct_dat["primer_desc"]

all_ints = list(set(ct_dat["primer_geno"]))
all_samps = list(set(ct_dat["sample_name"]))
samp_rands = list(set(ct_dat["samp_rand_name"]))
rep_rands = list(set(ct_dat["rep_rand_name"]))
primer_rands = list(set(ct_dat["primer_rand_name"]))
samp_primers = list(set(ct_dat["samp_primer"]))

ct_dat["int_ind"] = [all_ints.index(x) for x in ct_dat["primer_geno"]]
ct_dat["sample_ind"] = [all_samps.index(x) for x in ct_dat["sample_name"]]
ct_dat["samp_rand_ind"] = [samp_rands.index(x) for x in ct_dat["samp_rand_name"]]
ct_dat["rep_rand_ind"] = [rep_rands.index(x) for x in ct_dat["rep_rand_name"]]
ct_dat["primer_rand_ind"] = [primer_rands.index(x) for x in ct_dat["primer_rand_name"]]
ct_dat["samp_primer_ind"] = [samp_primers.index(x) for x in ct_dat["samp_primer"]]

distinct_geno_mask = ~ct_dat.duplicated("samp_primer_ind")
print(distinct_geno_mask)
distinct_geno_inds = np.where(distinct_geno_mask)[0]
print(ct_dat[["sample_name","primer_desc"]])
        
# set up a model to fit the primer efficiency for each primer
this_model=pm.Model()

with this_model:

    # we define the model in two pieces: first, the fits for primer efficiency, and second, the fits to get the ddCt values that we want
    # the primer efficiency part begins here:

    ## here is what we want to fit -- the primer efficiency of each primer pair
    ## note that we also have an intercept for each primer, but we care less about this

    log2_primer_effs = pm.Normal(
        'log2_primer_effs',
        mu=1,
        sigma=0.2,
        shape=len(all_primers),
    )
    primer_intercepts = pm.Uniform(
        'primer_intercepts',
        lower=-100,
        upper=100,
        shape=len(all_primers),
    )

    ct_sigma = pm.HalfCauchy( 'ct_sigma', beta=5 )
    ct_nu = pm.InverseGamma('ct_nu', alpha=0.5, beta=20.0)
    if with_dil_rands:
        dil_raneff_sigma = pm.HalfNormal('dil_raneff_sigma', sigma=2)
        dil_rand_effs = pm.Normal(
            'dil_rand_effs',
            mu=0,
            sigma=dil_raneff_sigma,
            shape=len(dil_rands),
        )
 
    ## now connect the observed Ct values with those parameters
    all_dilutions = -1 * dil_dat.log2dil.to_numpy()
    dil_cts = dil_dat.Cq.to_numpy()
    all_primer_inds = dil_dat.primer_inds.to_numpy()
    dil_rand_inds = dil_dat.dil_rand_ind.to_numpy()

    dil_mu = (
        primer_intercepts[all_primer_inds]
        + log2_primer_effs[all_primer_inds] * all_dilutions
    )

    if with_dil_rands:
        dil_mu += dil_rand_effs[dil_rand_inds]

    dil_Ct_vals = pm.StudentT(
        'dil_Ct_vals',
        mu=dil_mu,
        observed=dil_cts,
        sigma=ct_sigma,
        nu=ct_nu,
    )

    # and now here is the part to fit the delta Cts

    # in general, we assume that the Ct for each well has the form:
    # Ct = intercept(primer) + intercept(sample) + pulldown(primer) + pulldown:condition(primer) + (1|sample_repid)
    # I may want to add in additional random effect terms?
    # note that the primer based terms hold over from the part of the model noted above

    # here are the additional parameters we need to fit
    sample_intercepts = pm.Uniform('sample_intercepts', lower=-100, upper=100, shape=len(all_samps))
    pulldown_effs = pm.Normal('pulldown_effs', mu=0, sigma=5, shape=len(all_primers))
    interaction_effs = pm.Normal('interaction_effs', mu=0, sigma=5, shape=len(all_ints))
    ## and the random effect parameters
    samp_raneff_sigma = pm.HalfNormal('samp_raneff_sigma', sigma=2)
    samp_rand_effs = pm.Normal(
        'samp_rand_effs',
        mu=0,
        sigma=samp_raneff_sigma,
        shape=len(samp_rands),
    )
    if with_samp_primer_rands:
        primer_raneff_sigma = pm.HalfNormal('primer_raneff_sigma', sigma=2)
        primer_rand_effs = pm.Normal(
            'primer_rand_effs',
            mu=0,
            sigma=primer_raneff_sigma,
            shape=len(primer_rands),
        )

    # and here are the actual definitions of fitting observables

    ct_sample_inds = ct_dat.sample_ind.to_numpy()
    ct_primer_inds = ct_dat.primer_inds.to_numpy()
    ct_is_pulldown = 1.0 * (ct_dat["type"].to_numpy() == "extract")
    ct_is_not_wt = 1.0 * (ct_dat["geno"].to_numpy() != "py79")
    ct_int_inds = ct_dat.int_ind.to_numpy()
    ct_samp_rand_inds = ct_dat.samp_rand_ind.to_numpy()
    ct_primer_rand_inds = ct_dat.primer_rand_ind.to_numpy()
    samp_cts = ct_dat.Cq.to_numpy()

    ct_geno_mu = (
        primer_intercepts[ct_primer_inds]
        + sample_intercepts[ct_sample_inds]
        + ct_is_pulldown * pulldown_effs[ct_primer_inds]
        + ct_is_not_wt * ct_is_pulldown * interaction_effs[ct_int_inds]
    )

    if with_samp_primer_rands:
        ct_geno_mu += primer_rand_effs[ct_primer_rand_inds]

    ct_samp_mu = ct_geno_mu + samp_rand_effs[ct_samp_rand_inds]
    samp_Ct_vals = pm.StudentT(
        'samp_Ct_vals',
        mu=ct_samp_mu,
        sigma=ct_sigma,
        nu=ct_nu,
        observed=samp_cts,
    )

with this_model:
    #trace = pm.sample( 50, chains=1, tune=25, target_accept=0.95, backend="JAX" )
    trace = pm.sample( 5000, chains=n_chains, cores=n_chains, tune=2500, target_accept=0.8, backend="JAX" )
    pm.sample_posterior_predictive(trace, extend_inferencedata=True)

with this_model:
    az.summary(trace).to_csv(outprefix + "_summary.txt")
    trace.to_netcdf(outprefix + "_samples.cdf")
    with open(outprefix + "parameter_names.txt", "w") as ostr:
        for i,x in enumerate( all_samps ):
            ostr.write("sample_intercepts[% 3i] = %s\n" % (i,x))

        for i,x in enumerate( all_ints ):
            ostr.write("interaction_effs[% 3i] = %s\n" % (i,x))

        for i,x in enumerate( all_primers):
            ostr.write("primer_intercepts[% 3i] = %s\n" % (i,x))

        for i,x in enumerate( all_primers):
            ostr.write("pulldown_effs[% 3i] = %s\n" % (i,x))

        for i,x in enumerate( all_primers):
            ostr.write("log2_primer_effs[% 3i] = %s\n" % (i,x))

    for param in [
            ["log2_primer_effs", "primer_intercepts"], ["dil_rand_effs"],
            ["sample_intercepts", "pulldown_effs", "interaction_effs", "samp_raneff_sigma","samp_rand_effs"]
    ]:
        plt.figure()
        az.plot_trace(trace, var_names=param)
        param_str = "_".join(param)
        plt.savefig(outprefix + f'_{param_str}_trace.png')
     
