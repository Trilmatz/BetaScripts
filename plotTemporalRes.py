import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uncertainties import unumpy, ufloat_fromstr

def plot_temporal_resolution(prefix, temperature=20):
    filepath=f"plots/{prefix}/sigmas.csv"
    sigmas_df = pd.read_csv(filepath, index_col=0)
    sigmas_df = sigmas_df.map(ufloat_fromstr)
    print("Loaded temporal resolution data:")
    print(sigmas_df)

    biases = sigmas_df.index
    labels = sigmas_df.columns

    # plt.figure(figsize=(10, 6))
    for label in labels[:]:
        plt.errorbar(
            biases, unumpy.nominal_values(sigmas_df[label]), yerr=unumpy.std_devs(sigmas_df[label]),
            fmt="o", label=f"{label}")
    # plt.plot(sigmas)
    plt.xlabel("Bias [V]")
    plt.ylabel(fr"$\sigma$ [ps]")
    plt.legend(title=rf"$T = {temperature}\,$°C")
    plt.ylim(0, 70)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/{prefix}/temporal_res.pdf")
    plt.close()

def compare_resolutions():
    temperatures = [20, -20]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    plt.figure(figsize=(10, 6))
    for temperature, marker, line, fillstyle in zip(temperatures, ["s", "D"], ["-", "--"], ["full", "none"]):
        filepath=f"plots/{temperature}C/sigmas.csv"
        sigmas_df = pd.read_csv(filepath, index_col=0)
        sigmas_df = sigmas_df.map(ufloat_fromstr)
        biases = sigmas_df.index
        if temperature == -20:
            biases += 40
        labels = sigmas_df.columns[0:2]
        for label, color in zip(labels, colors):
            plt.errorbar(
                biases, unumpy.nominal_values(sigmas_df[label]), yerr=unumpy.std_devs(sigmas_df[label]),
                marker=marker, fillstyle=fillstyle, linestyle=line, color=color)
    plt.plot([], [], color="black", marker="s", linestyle="-", label="20°C")
    plt.plot([], [], color="black", marker="D", linestyle="--", label="-20°C", fillstyle="none")
    for label, color in zip(labels, colors):
        plt.plot([], [], color=color, label=f"Channel {label}")
    plt.xlabel("Bias [V]")
    plt.ylabel(fr"$\sigma$ [ps]")
    plt.legend()
    plt.ylim(0, 80)
    plt.tight_layout()
    plt.savefig(f"plots/temporal_res_comparison.pdf")
    plt.close()


if __name__ == "__main__":
    plot_temporal_resolution(prefix="CNM_W4_C18")
    # plot_temporal_resolution(-20)
    # compare_resolutions()