import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotTemporalRes import plot_temporal_resolution
from plotCharges import plot_charges
from uncertainties import unumpy, ufloat_fromstr

def plot_values(df, temperature=20):
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
    plt.legend(title=rf"$T = {temperature}\,$°C")
    plt.tight_layout()
    plt.savefig(f"plots/{temperature}C/values.pdf")
    plt.close()

def plot_jitters(temperature=20):
    filepath=f"plots/{temperature}C/jitter.csv"
    jitters_df = pd.read_csv(filepath, index_col=0)
    jitters_df = jitters_df.map(ufloat_fromstr)
    print("Loaded jitter data:")
    print(jitters_df)

    biases = jitters_df.index
    labels = jitters_df.columns

    plt.figure(figsize=(10, 6))
    for label in labels:
        plt.errorbar(
            biases, unumpy.nominal_values(jitters_df[label]), yerr=unumpy.std_devs(jitters_df[label]),
            fmt="o", label=f"{label}")
        # plt.plot(biases, jitters_df[label], 'o', label=f"{label}")
    # plt.plot(sigmas)
    plt.xlabel("Bias [V]")
    plt.ylabel(r"$\sigma_{{\text{jitter}}} = N / (dV/dt)$ [ps]")
    plt.legend(title=rf"$T = {temperature}\,$°C")
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f"plots/{temperature}C/jitters.pdf")
    plt.close()


if __name__ == "__main__":
    temperature = 20
    plot_temporal_resolution(temperature=temperature)
    plot_charges(temperature=temperature)
    plot_jitters(temperature=temperature)
