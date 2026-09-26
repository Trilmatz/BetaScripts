import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uncertainties import unumpy, ufloat_fromstr

def plot_data(df, labels=None, names=None):
    biases = df.index

    # plt.figure(figsize=(10, 6))
    if labels is None:
        labels = df.columns
    if names is None:
        names = labels
    for label, name in zip(labels, names):
        try:
            plt.errorbar(
                biases, unumpy.nominal_values(df[label]), yerr=unumpy.std_devs(df[label]),
                fmt="o", label=name)
        except Exception:
            plt.plot(biases, df[label], 'o', label=name)
    # plt.plot(sigmas)


if __name__ == "__main__":
    file = "charges.csv"
    labels = ["1"]
    df1 = pd.read_csv(f"plots/CNM_W4_C18/{file}", index_col=0)
    df2 = pd.read_csv(f"plots/CNM_W4_H21/{file}", index_col=0)

    for df, prefix in zip([df1, df2], ["CNM_W4_C18", "CNM_W4_H21"]):
        try:
            df = df.map(ufloat_fromstr)
        except Exception as e:
            pass
        plot_data(df, labels=labels, names=[prefix])
    plt.xlabel("Bias [V]")
    plt.ylabel(fr"$Q$ [fC]")
    plt.legend(title=rf"$T = 20\,$°C")
    # plt.ylim(0, 35)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/charges_CNM_W4.pdf")
    plt.close()

    Q_pin = 0.5
    for df, prefix in zip([df1, df2], ["CNM_W4_C18", "CNM_W4_H21"]):
        biases = df.index
        plt.plot(biases, df["1"]/Q_pin, 'o', label=prefix) # shifting H21 by -2.5V aligns the curves
    plt.xlabel("Bias [V]")
    plt.ylabel(fr"Gain")
    plt.legend(title=rf"$Q_{{\rm pin}} = {Q_pin}\,$fC")
    # plt.ylim(0, 35)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/gain_CNM_W4.pdf")
    plt.close()

    file = "sigmas.csv"
    labels = ["hyp"]
    res_1 = pd.read_csv(f"plots/CNM_W4_C18/{file}", index_col=0)
    res_2 = pd.read_csv(f"plots/CNM_W4_H21/{file}", index_col=0)

    for df, res, prefix in zip([df1, df2], [res_1, res_2], ["CNM_W4_C18", "CNM_W4_H21"]):
        res = res.map(ufloat_fromstr)

        plt.errorbar(unumpy.nominal_values(df["1"])/Q_pin, unumpy.nominal_values(res["hyp"]), yerr=unumpy.std_devs(res["hyp"]), fmt='o', label=prefix)

    plt.xlabel("Gain")
    plt.ylabel(fr"Temporal resolution [ps]")
    plt.legend(title=rf"$Q_{{\rm pin}} = {Q_pin}\,$fC")
    plt.ylim(0, 70)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"plots/res_gain_CNM_W4.pdf")
    plt.close()