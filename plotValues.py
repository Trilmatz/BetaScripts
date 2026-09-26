import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotTemporalRes import plot_temporal_resolution
from plotCharges import plot_charges
from plotJitters import plot_jitters
from plotNoise import plot_noise
from uncertainties import unumpy, ufloat_fromstr

def plot_values(df, temperature=20):
    biases = df.index
    labels = df.columns

    # plt.figure(figsize=(10, 6))
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
    plt.legend(title=rf"$T = {temperature}\,$°C")
    plt.tight_layout()
    plt.savefig(f"plots/{temperature}C/values.pdf")
    plt.close()


if __name__ == "__main__":
    temperature = 20
    prefix = "CNM_W4_C18"
    plot_temporal_resolution(prefix)
    plot_charges(prefix)
    plot_jitters(prefix)
    plot_noise(prefix)
