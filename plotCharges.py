import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uncertainties import unumpy, ufloat_fromstr

def plot_charges(temperature=20):
    filepath=f"plots/{temperature}C/charges.csv"
    charges_df = pd.read_csv(filepath, index_col=0)
    # charges_df = charges_df.map(ufloat_fromstr)
    print("Loaded charge data:")
    print(charges_df)

    biases = charges_df.index
    labels = charges_df.columns

    plt.figure(figsize=(10, 6))
    for label in labels:
        # plt.errorbar(
        #     biases, unumpy.nominal_values(charges_df[label]), yerr=unumpy.std_devs(charges_df[label]),
        #     fmt="o", label=f"{label}")
        plt.plot(biases, charges_df[label], 'o', label=f"{label}")
    # plt.plot(sigmas)
    plt.xlabel("Bias [V]")
    plt.ylabel(fr"$Q$ [fC]")
    plt.legend(title=rf"$T = {temperature}\,$°C")
    plt.ylim(0, 80)
    plt.tight_layout()
    plt.savefig(f"plots/{temperature}C/charges.pdf")
    plt.close()

def compare_resolutions():
    temperatures = [20, -20]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    plt.figure(figsize=(10, 6))
    for temperature, marker, line, fillstyle in zip(temperatures, ["s", "D"], ["-", "--"], ["full", "none"]):
        filepath=f"plots/{temperature}C/charges.csv"
        charges_df = pd.read_csv(filepath, index_col=0)
        charges_df = charges_df.map(ufloat_fromstr)
        biases = charges_df.index
        if temperature == -20:
            biases += 40
        labels = charges_df.columns[0:2]
        for label, color in zip(labels, colors):
            plt.errorbar(
                biases, unumpy.nominal_values(charges_df[label]), yerr=unumpy.std_devs(charges_df[label]),
                marker=marker, fillstyle=fillstyle, linestyle=line, color=color)
    plt.plot([], [], color="black", marker="s", linestyle="-", label="20°C")
    plt.plot([], [], color="black", marker="D", linestyle="--", label="-20°C", fillstyle="none")
    for label, color in zip(labels, colors):
        plt.plot([], [], color=color, label=f"Channel {label}")
    plt.xlabel("Bias [V]")
    plt.ylabel(fr"$Q$ [fC]")
    plt.legend()
    plt.ylim(0, 80)
    plt.tight_layout()
    plt.savefig(f"plots/charges_comparison.pdf")
    plt.close()


if __name__ == "__main__":
    plot_charges(20)
    plot_charges(-20)
    compare_resolutions()