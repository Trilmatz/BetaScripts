import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uncertainties import unumpy, ufloat_fromstr

def plot_jitters(prefix, temperature=20):
    filepath=f"plots/{prefix}/jitters.csv"
    jitters_df = pd.read_csv(filepath, index_col=0)
    jitters_df = jitters_df.map(ufloat_fromstr)
    print("Loaded jitter data:")
    print(jitters_df)

    biases = jitters_df.index
    labels = jitters_df.columns

    # plt.figure(figsize=(5,3))
    for label in labels:
        plt.errorbar(
            biases, unumpy.nominal_values(jitters_df[label]), yerr=unumpy.std_devs(jitters_df[label]),
            fmt="o", label=f"{label}")
        # plt.plot(biases, jitters_df[label], 'o', label=f"{label}")
    # plt.plot(sigmas)
    plt.xlabel("Bias [V]")
    plt.ylabel(fr"$\sigma_{{\text{{jitter}}}} = N / (dV/dt)$ [ps]")
    plt.legend(title=rf"$T = {temperature}\,$°C")
    plt.ylim(0, 40)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/{prefix}/jitters.pdf")
    plt.close()

def plot_landau(prefix, temperature=20):
    filepath=f"plots/{prefix}/jitters.csv"
    jitters_df = pd.read_csv(filepath, index_col=0)
    jitters_df = jitters_df.map(ufloat_fromstr)
    print(jitters_df)

    filepath=f"plots/{prefix}/sigmas.csv"
    sigmas_df = pd.read_csv(filepath, index_col=0)
    sigmas_df = sigmas_df.map(ufloat_fromstr)
    print(sigmas_df)

    biases = jitters_df.index
    landaus = unumpy.sqrt(sigmas_df["hyp"]**2 - jitters_df["1"]**2)
    print(landaus)

    plt.figure(figsize=(5, 3))
    
    plt.errorbar(
        biases, unumpy.nominal_values(landaus), yerr=unumpy.std_devs(landaus),
        fmt="o", label=f"1")

    plt.xlabel("Bias [V]")
    plt.ylabel(fr"$\sigma_{{\text{{landau}}}} = \sqrt{{\sigma^2 - \sigma_{{jitter}}^2}}$ [ps]")
    # plt.legend(title=rf"$T = {temperature}\,$°C")
    plt.ylim(0, 40)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/{prefix}/landau.pdf")
    plt.close()

if __name__ == "__main__":
    plot_jitters("CNM_W4_C18")
    plot_landau("CNM_W4_C18")