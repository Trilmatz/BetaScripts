import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BetaAnalisi import Analisi
from file_mapping import bias_info
from plotTemporalRes import plot_temporal_resolution
from plotCharges import plot_charges
from plotJitters import plot_jitters, plot_landau
from uncertainties import unumpy

temperature = 20
if temperature == -20:
    biases = [130, 150, 170, 175, 180]
elif temperature == 20:
    biases = [150, 170, 190, 210, 215, 220, 225]

prefix = "CNM_W4_C18"
biases = [220, 240, 260, 270, 275, 280, 285]

# biases = [220]
sigmas = []
sizes = []
charges = []
rms_values = []
jitters = []

cfd_val = 0.3

# bias_info = bias_info[temperature]
bias_info = bias_info[prefix]

def plot_values(df, prefix, temperature=20):
    biases = df.index
    labels = df.columns

    plt.figure(figsize=(10, 6))
    for label in labels:
        try:
            plt.errorbar(
                biases, unumpy.nominal_values(df[label]), yerr=unumpy.std_devs(df[label]),
                fmt="o", label=f"{label}")
        except Exception:
            plt.plot(biases, df[label], 'o', label=f"{label}")
    # plt.plot(sigmas)
    plt.xlabel("Bias [V]")
    plt.ylabel(fr"Value [a.u.]")
    # plt.legend(title=rf"$T = {temperature}\,$°C")
    plt.tight_layout()
    plt.savefig(f"plots/{prefix}/values.pdf")
    plt.close()


for bias in biases:
    path = "CNM/W4/C18"
    analyzer = Analisi(bias_info[bias]["name"], f"data/{path}/{bias}V", f"plots/{prefix}/{bias}V")

    analyzer.load_data()
    size = analyzer.apply_cuts(bias_info[bias]["cuts"], max_thresholds=bias_info[bias]["upper_cuts"])
    sizes.append(size)

    analyzer.plot_amplitude_distribution()
    sigma_bias = analyzer.analyze_temporal_resolution(cfd_val=cfd_val, res_timeref=33.3)
    sigmas.append(sigma_bias)

    analyzer.plot_wfm_cut_validation(channel=0, num_events=200)
    analyzer.plot_waveforms(amplitude_threshold=700)
    analyzer.plot_snr(cfd_val=cfd_val)
    analyzer.plot_cfd(cfd_val=cfd_val)

    charge = analyzer.plot_charge()
    charges.append(charge)

    rms = analyzer.calculate_rms()
    rms_values.append(rms)
    jitter = analyzer.plot_jitter()
    jitters.append(jitter)


biases = np.array(biases)
sigmas = np.array(sigmas)
charges = np.array(charges)
rms_values = np.array(rms_values)
jitters = np.array(jitters)

labels = ("hyp", "1", "2", "3")
sigmas_df = pd.DataFrame(sigmas, columns=labels, index = biases)
sigmas_df.index.name = "bias"
sigmas_df.to_csv(f"plots/{prefix}/sigmas.csv")

labels = ("1", "2", "3")
charges_df = pd.DataFrame(charges, columns=labels, index = biases)
charges_df.index.name = "bias"
charges_df.to_csv(f"plots/{prefix}/charges.csv")

noise_df = pd.DataFrame(rms_values, columns=labels, index = biases)
noise_df.index.name = "bias"
noise_df.to_csv(f"plots/{prefix}/noise.csv")

jitter_df = pd.DataFrame(jitters, columns=labels, index = biases)
jitter_df.index.name = "bias"
jitter_df.to_csv(f"plots/{prefix}/jitters.csv")


plot_temporal_resolution(prefix=prefix, temperature=temperature)
plot_charges(prefix=prefix, temperature=temperature)
plot_jitters(prefix=prefix, temperature=temperature)
plot_landau(prefix=prefix)
# plot_values(df, prefix=prefix, temperature=temperature)
