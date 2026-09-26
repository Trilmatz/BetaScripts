import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uncertainties import unumpy, ufloat_fromstr

def plot_charges(prefix, temperature=20):
    filepath=f"plots/{prefix}/charges.csv"
    charges_df = pd.read_csv(filepath, index_col=0)
    # charges_df = charges_df.map(ufloat_fromstr)
    print("Loaded charge data:")
    print(charges_df)

    biases = charges_df.index
    columns = charges_df.columns
    
    if prefix[-1] == "P":
        columns = columns[:1]
    labels = {
        "1": "Data",
        "2": "HPK timeref",
        "3": "HPK trigger"
    }
    # plt.figure(figsize=(10, 6))
    for column in columns:
        # plt.errorbar(
        #     biases, unumpy.nominal_values(charges_df[label]), yerr=unumpy.std_devs(charges_df[label]),
        #     fmt="o", label=f"{label}")
        plt.plot(biases, charges_df[column], 'o', label=f"{labels[column]}")
        if prefix[-1] == "P":
            plt.axhline(y=0.55, color="red", linestyle="-.", label=r"$W_{\rm pin} = 0.55\,\rm\mu$m")
            plt.axhline(y=charges_df[column].mean(), color='tab:green', label=fr"$Q_{{\rm avg}}={charges_df[column].mean():.2f}\,$fC")
            plt.axhline(y=0.45, color="red", linestyle="--", label=r"$W_{\rm pin} = 0.45\,\rm\mu$m")

    if prefix[-1] == "P":
        plt.ylim(0, 1)
    else:
        plt.ylim(0, 80)
    # plt.plot(sigmas)
    plt.xlabel("Bias [V]")
    plt.ylabel(fr"$Q$ [fC]")
    plt.legend(title=rf"$T = {temperature}\,$°C")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/{prefix}/charges.pdf")
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
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/charges_comparison.pdf")
    plt.close()


if __name__ == "__main__":
    plot_charges("CNM_W4_F19P", temperature=20)
    #plot_charges(-20)
    #compare_resolutions()