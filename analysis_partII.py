import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns
import re
import glob

def summarize(res):
    summary_geno = []
    summary_locus = []
    summary_mean = []
    summary_median = []
    summary_sd = []
    mean = np.mean(res, axis=2)
    median = np.median(res, axis=2)
    stdev = np.std(res, axis=2)
    for (i,geno) in enumerate(genotypes):
        for (j,locus) in enumerate(loci):
            summary_geno.append(geno)
            summary_locus.append(locus)
            summary_mean.append(mean[i,j])
            summary_median.append(median[i,j])
            summary_sd.append(stdev[i,j])

    out_summary = pd.DataFrame({
        "genotype": summary_geno,
        "locus": summary_locus,
        "mean": summary_mean,
        "median": summary_median,
        "stdev": summary_sd,
    })
    return out_summary

def arviz_summarize(res):
    
    summary_geno = []
    summary_locus = []
    summary_mean = []
    summary_lower = []
    summary_upper = []
    for (i,geno) in enumerate(genotypes):
        for (j,locus) in enumerate(loci):
            summary_geno.append(geno)
            summary_locus.append(locus)
            dat = az.convert_to_inference_data(res[i,j,:])
            dat_summary = az.summary(dat, hdi_prob=0.95)
            summary_mean.append(dat_summary["mean"][0])
            summary_lower.append(dat_summary["hdi_2.5%"][0])
            summary_upper.append(dat_summary["hdi_97.5%"][0])

    out_summary = pd.DataFrame({
        "genotype": summary_geno,
        "locus": summary_locus,
        "mean": summary_mean,
        "lower": summary_lower,
        "upper": summary_upper,
    })
    return out_summary

infile_qpcr = "long_form_table.csv"
infile_quant = "qPCR_picogram.csv"
count_files = glob.glob("*strand_col_qpcr_loci.txt")
count_genos = []
count_reps = []
count_types = []
counts = []
loci = []
for count_file in count_files:
    split_name = count_file.split("_")
    geno = split_name[0]
    rep = int(split_name[1][-1])
    ext = split_name[2]
    if ext == "hbd":
        the_type = "extract"
    else:
        the_type = "input"
    with open(count_file, "r") as f:
        for line in f:
            #print(line)
            e = line.strip().split("\t")
            count = e[-1]
            locus = e[-2].split(";")[1]
            if "pseudo" in locus:
                locus = "pseudo"
            elif locus in ["sdpC", "mfd", "BraB", "bsrI"]:
                locus += "_upst"
            elif locus == "upstCspB":
                locus = "cspB_upst"
            elif locus == "arok":
                locus = "aroK_upst"
            locus = "Bs_" + locus
            count_genos.append(geno)
            count_reps.append(rep)
            count_types.append(the_type)
            counts.append(float(count))
            loci.append(locus)
 
counts_df = pd.DataFrame({
    "genotype": count_genos,
    "replicate": count_reps,
    "sample_type": count_types,
    "locus": loci,
    "count": counts,
})
print(counts_df)
counts_df.to_csv("counts_df.csv")

outprefix = "results"
infile = "fit_effs_dil_rands_samples.cdf"
param_summary_file = "fit_effs_dil_rands_summary.txt"
lookup_file = "fit_effs_dil_randsparameter_names.txt"
out_summary_file = outprefix + "_summary.csv"

quant_tab = pd.read_csv( infile_quant )
qpcr_tab = pd.read_csv( infile_qpcr )
ct_dat = qpcr_tab[np.logical_or(qpcr_tab.type == "extract", qpcr_tab.type == "input")].copy()

dil_dat = qpcr_tab[
        qpcr_tab.type == "dill_ser"
    ].copy(
    ).merge(quant_tab, how="left", on=["sample_name","type","bio_Rep","geno","relative_concentration"])
#print(dil_dat)
dil_dat = dil_dat[~np.isnan(dil_dat["picograms"])].copy()
all_primers = list(set(dil_dat["primer_desc"]))
ct_dat["primer_inds"] = [all_primers.index(x) for x in ct_dat["primer_desc"] ]

## also figure out indexing for other factors that enter into the model
ct_dat["samp_rand_name"] = ct_dat["sample_name"] + ct_dat["date"]
ct_dat["primer_rand_name"] = ct_dat["primer_desc"] + ct_dat["date"]
ct_dat["primer_geno"] = ct_dat["primer_desc"] + "_" + ct_dat["geno"]
ct_dat["samp_primer"] = ct_dat["sample_name"] + ct_dat["primer_desc"]

all_ints = list(set(ct_dat["primer_geno"]))
all_samps = list(set(ct_dat["sample_name"]))
samp_rands = list(set(ct_dat["samp_rand_name"]))
primer_rands = list(set(ct_dat["primer_rand_name"]))
samp_primers = list(set(ct_dat["samp_primer"]))

ct_dat["int_ind"] = [all_ints.index(x) for x in ct_dat["primer_geno"]]
ct_dat["sample_ind"] = [all_samps.index(x) for x in ct_dat["sample_name"]]
ct_dat["samp_rand_ind"] = [samp_rands.index(x) for x in ct_dat["samp_rand_name"]]
ct_dat["primer_rand_ind"] = [primer_rands.index(x) for x in ct_dat["primer_rand_name"]]
ct_dat["samp_primer_ind"] = [samp_primers.index(x) for x in ct_dat["samp_primer"]]

summary_df = pd.read_csv(param_summary_file)

lut = pd.read_csv(lookup_file, sep=" = ", names=["id","name"], engine="python")
lut[["param", "tmp"]] = lut["id"].str.split("\[", expand=True)
lut["idx"] = lut["tmp"].str.strip().str.rstrip("\]").astype("int")
lut["bact"] = lut["name"].str.split("_", expand=True)[0]
geno_betas = lut[lut["param"] == "interaction_effs"].copy()
#pe_df = lut[lut["param"] == "log2_primer_effs"].copy()
#print(lut)

name_list = list(geno_betas["name"])
geno_list = []
locus_list = []
for s in name_list:
    e = s.split("_")
    if len(e) == 4:
        locus = e[1] + "_" + e[2]
    else:
        locus = e[1]
    geno_list.append(e[-1])
    locus_list.append(locus)

geno_betas["geno"] = geno_list
geno_betas["locus"] = locus_list

#geno_betas = geno_betas[geno_betas["geno"] != "py79"].copy()
geno_betas["search"] = geno_betas["id"].str.replace(" ", "")

primer_betas = lut[lut["param"] == "pulldown_effs"].copy()
primer_betas["search"] = primer_betas["id"].str.replace(" ", "")

with_summary = primer_betas.merge(summary_df[["Unnamed: 0", "mean"]], how="left", left_on="search", right_on="Unnamed: 0")
sorted_cc = with_summary[with_summary["bact"] == "Cc"].sort_values(["mean"])
print(sorted_cc[["name","mean"]])
cc_ref_locus = sorted_cc["name"].iloc[0]
cc_ref_locus = cc_ref_locus.split("_")[-1]
print(cc_ref_locus)
cc_ref_idx = sorted_cc["idx"].iloc[0]
print(cc_ref_idx)

name_list = list(primer_betas["name"])
primer_locus_list = []
for s in name_list:
    e = s.split("_")
    if len(e) == 3:
        locus = e[1] + "_" + e[2]
    else:
        locus = e[1]
    primer_locus_list.append(locus)

primer_betas["locus"] = primer_locus_list

trace = az.from_netcdf(infile)
# remove burn-in and stack the chains
samps = trace.sel(draw=slice(2500,None))
summary = az.summary(samps, stat_focus="median").reset_index()
print(summary)
print(np.any(summary["r_hat"] > 1.02))
print(summary[summary["r_hat"] > 1.02])
summary["var_name"] = summary["index"].str.split("[", expand=True)[0]
summary["idx"] = summary["index"].str.split("[", expand=True)[1].str.strip("]")
samples = samps.stack(sample=["chain","draw"])

pulldown_effs = samples.posterior.pulldown_effs.as_numpy()
interaction_effs = samples.posterior.interaction_effs.as_numpy()
log2_primer_effs = samples.posterior.log2_primer_effs.as_numpy()

print(pulldown_effs.shape)
print(interaction_effs.shape)
print(log2_primer_effs.shape)

bsub_betas = geno_betas[geno_betas["bact"] == "Bs"].copy()
cc_betas = geno_betas[geno_betas["bact"] == "Cc"].copy()
bsub_primers = primer_betas[primer_betas["bact"] == "Bs"].copy()
cc_primers = primer_betas[primer_betas["bact"] == "Cc"].copy()
cc_wt_j_samples = pulldown_effs[cc_ref_idx,:]
# primer efficienties have same indexing at pulldow_effs
cc_pe_j_samples = log2_primer_effs[cc_ref_idx,:]

enrich_over_cc = []
genotypes = np.unique(bsub_betas["geno"])
loci = np.unique(bsub_betas["locus"])
results = np.zeros((len(genotypes),len(loci),pulldown_effs.shape[1]))
#results2 = np.zeros((len(genotypes),len(loci),pulldown_effs.shape[1]))
for (i,geno) in enumerate(genotypes):
    bsub_i = bsub_betas[bsub_betas["geno"] == geno]
    cc_i = cc_betas[cc_betas["geno"] == geno]
    #print(cc_i)
    cc_i_j = cc_i[cc_i["locus"] == cc_ref_locus]
    #bsub_pe_i = 

    if geno == "py79":
        cc_i_j_samples = 0
    else:
        cc_i_j_samples = interaction_effs[cc_i_j["idx"].iloc[0],:]
    for (j,locus) in enumerate(loci):
        bsub_i_j = bsub_i[bsub_i["locus"] == locus]
        if geno == "py79":
            bsub_i_j_samples = 0
        else:
            bsub_i_j_samples = interaction_effs[bsub_i_j["idx"].iloc[0],:]
        bsub_wt_j = bsub_primers[bsub_primers["locus"] == locus]
        bsub_wt_j_samples = pulldown_effs[bsub_wt_j["idx"].iloc[0],:]
        bsub_pe_j_samples = log2_primer_effs[bsub_wt_j["idx"].iloc[0],:]

        result = (
            (bsub_wt_j_samples + bsub_i_j_samples) * bsub_pe_j_samples
            - (cc_wt_j_samples + cc_i_j_samples) * cc_pe_j_samples
        )
        results[i,j,:] = result

np.save("bs_pulldown_relative_to_cc.npy", results)
out_summary = arviz_summarize(results)

out_summary.to_csv(out_summary_file)

