import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uncertainties import unumpy, ufloat_fromstr

def plot_noise(prefix, temperature=20):
    filepath=f"plots/{prefix}/noise.csv"
    jitters_df = pd.read_csv(filepath, index_col=0)
    jitters_df = jitters_df.map(ufloat_fromstr)
    print("Loaded noise data:")
    print(jitters_df)

    biases = jitters_df.index
    labels = jitters_df.columns

    # plt.figure(figsize=(10, 6))
    for label in labels:
        plt.errorbar(
            biases, unumpy.nominal_values(jitters_df[label]), yerr=unumpy.std_devs(jitters_df[label]),
            fmt="o", label=f"{label}")
        # plt.plot(biases, jitters_df[label], 'o', label=f"{label}")
    # plt.plot(sigmas)
    plt.xlabel("Bias [V]")
    plt.ylabel(fr"RMS noise [mV]")
    plt.legend(title=rf"$T = {temperature}\,$°C")
    plt.ylim(0, 5)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/{prefix}/noise.pdf")
    plt.close()

if __name__ == "__main__":
    plot_noise("CNM_W4_C18")