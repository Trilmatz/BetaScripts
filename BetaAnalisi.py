import os
import argparse
import uproot
import numpy as np
import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt
import uncertainties as unc
import scienceplots

from uncertainties.umath import sqrt
from landaupy import langauss
from scipy.optimize import curve_fit
from scipy.stats import landau, norm, t

# plt.style.use('science')

class Analisi:
    """
    A class to handle loading, cutting, and plotting ROOT waveform data 
    using uproot, awkward, and pandas.
    """
    
    def __init__(self, filename, bias, temperature=20, tree_name="Analysis"):
        self.filepath = f"data/{temperature}C/{bias}V/{filename}"
        self.tree_name = tree_name
        self.temperature = temperature

        self.save_dir = f"plots/{temperature}C/{bias}V"
        
        # Load the ROOT file and tree
        self.file = uproot.open(self.filepath)
        self.tree = self.file[self.tree_name]
        
        # Data containers
        self.raw = {}
        self.passed = {}
        self.rejected = {}

        print(f"\n{'='*50}\nInitialized object to analyse measurement of {bias}V stored in {filename} at {temperature}C.\n{'='*50}\n")


    @staticmethod
    def gaussian(x, amplitude, mean, sigma):
        return amplitude * norm.pdf(x, loc=mean, scale=sigma)

    @staticmethod
    def fit_langauss(counts, edges):
        bin_width = edges[1] - edges[0]
        amplitude = np.sum(counts) * bin_width
        centers = 0.5 * (edges[:-1] + edges[1:])
        initial_guess = [centers[np.argmax(counts)], 3.0, 2.0]
        bounds = ([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf])
        sigma = np.where(counts > 0, np.sqrt(counts), 1.0) / amplitude
        try:
            popt, pcov = curve_fit(
                langauss.pdf,
                centers,
                counts / amplitude,
                p0=initial_guess,
                sigma = sigma,
                absolute_sigma=True,
                bounds=bounds,
                maxfev=20000,
            )

            x_fit = np.linspace(edges[0], edges[-1], 500)
            y_fit = amplitude * langauss.pdf(x_fit, *popt)

            peak_index = np.argmax(y_fit)
            mpv = np.round(x_fit[peak_index], 2)
        except (RuntimeError, ValueError):
            mpv = 0
            popt = [0, 0, 0]
            pcov = np.zeros((3, 3))
            amplitude = 0
        return mpv, popt, pcov, x_fit, y_fit, amplitude

    def fit_gauss(self, counts, edges):
        bin_width = edges[1] - edges[0]
        amplitude = np.sum(counts) * bin_width
        centers = 0.5 * (edges[:-1] + edges[1:])
        initial_guess = [amplitude, 0, 5]
        bounds = ([0.0, -np.inf, 0.0], [np.inf, np.inf, np.inf])
        sigma = np.where(counts > 0, np.sqrt(counts), 1.0)
        try:
            popt, pcov = curve_fit(
                self.gaussian,
                centers,
                counts,
                p0=initial_guess,
                sigma = sigma,
                absolute_sigma=True,
                bounds=bounds,
                maxfev=20000,
            )

            x_fit = np.linspace(edges[0], edges[-1], 500)
            correlated_params = unc.correlated_values(popt, pcov)
            sigmas_ufloat = correlated_params[2]
            y_fit = self.gaussian(x_fit, *popt)

        except (RuntimeError, ValueError):
            sigmas_ufloat = unc.ufloat(0, 1, tag="mV")
            popt = [0, 0, 0]
            pcov = np.zeros((3, 3))
        return sigmas_ufloat, popt, pcov, x_fit, y_fit


    @staticmethod
    def student_t(x, amplitude, mean, sigma, nu):
        return amplitude * t.pdf(x, df=nu, loc=mean, scale=sigma)


    def load_data(self):
        """Loads branches into 2D and 3D NumPy arrays."""
        # Add 'rms' or 'cfd' to this list if you need them later
        branches = self.tree.arrays(["w", "t", "pmax", "negpmax", "event", "cfd", "rms", "area_new", "dvdt_2080"])
        
        # Waveforms [events, channels, samples]
        self.raw['w'] = ak.to_numpy(branches["w"]) * 1e3  # in mV
        self.raw['t'] = ak.to_numpy(branches["t"]) * 1e9  # in ns

        # cfd values [events, channels, [time at 10%, 20%, 30%, 40%, 50%, 60%, 70%]]
        self.raw['cfd'] = ak.to_numpy(branches["cfd"]) # in ns already
        
        # Amplitudes [events, channels]
        self.raw['pmax'] = ak.to_numpy(branches["pmax"])
        self.raw['negpmax'] = ak.to_numpy(branches["negpmax"])
        self.raw['rms'] = ak.to_numpy(branches["rms"])
        self.raw['area_new'] = ak.to_numpy(branches["area_new"])
        self.raw['dvdt_2080'] = ak.to_numpy(branches["dvdt_2080"])

        # Event metadata [events]
        self.raw['event'] = ak.to_numpy(branches["event"])
        
        print(f"Loaded {self.raw['w'].shape[0]} events across {self.raw['w'].shape[1]} channels.")

    def apply_cuts(self, min_thresholds, max_threshold=800, observable="pmax", cut_rms=None):
        """
        Creates a boolean mask for the cuts and splits the dataset into 
        'passed' and 'rejected' events across all variables.
        """
        if not self.raw:
            self.load_data()
            
        obs = self.raw[observable]
        
        # 1. Upper limit mask: All channels must be below max_threshold
        upper_mask = np.all(obs < max_threshold, axis=1)
        
        # 2. Lower limit mask: Each channel must be above its respective minimum threshold
        min_thresh_array = np.array(min_thresholds)
        lower_mask = np.all(obs > min_thresh_array, axis=1)
        
        # The final mask requires both conditions to be true
        master_mask = upper_mask & lower_mask
        
        # Apply the mask (passed) and inverted mask (rejected) to ALL data arrays simultaneously
        for key, array in self.raw.items():
            self.passed[key] = array[master_mask]
            self.rejected[key] = array[~master_mask]

        if cut_rms is not None:
            initial_passed_count = self.passed['w'].shape[0]
            rms_mask = np.all(self.passed['rms'] < cut_rms, axis=1)
            for key in self.passed.keys():
                self.passed[key] = self.passed[key][rms_mask]
            print(f"Applied RMS cut: {cut_rms} mV. Removed events: {self.passed['w'].shape[0]- initial_passed_count}.")
            
        print(f"Passed: {np.sum(master_mask)} | Rejected: {np.sum(~master_mask)} | Total Events: {len(obs)}")
        return np.sum(master_mask), len(obs)


    def calculate_rms(self, idx=10, save_plot=True):
        """Calculates RMS for a specific channel."""
        self.passed['noise'] = np.zeros((self.passed['w'].shape[0], 3))
        for channel in range(3):
            self.passed['noise'][:, channel] = self.passed['w'][:, channel, idx]

        os.makedirs(self.save_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        rms_values = []
        
        for ch in range(3):
            ax = axes[ch]
            counts, edges, _ = ax.hist(self.passed['noise'][:, ch], bins=50, range=(-15, 15), color="tab:blue", alpha=0.7, label="Data")
            sigma_ufloat, popt, pcov, fit_x, fit_y = self.fit_gauss(counts, edges)
            ax.plot(fit_x, fit_y, linewidth=2, label=fr"Fit: $\sigma={sigma_ufloat}\,$mV", color="red")
            rms_values.append(sigma_ufloat)
            ax.set_xlabel(rf"Signal at idx {idx} CH{ch+1} [mV]")
            ax.set_ylabel("Counts")
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        if save_plot:
            plt.savefig(os.path.join(self.save_dir, "noise_distribution.pdf"))
        plt.close()
        return rms_values

    # def plot_rms(self):
    #     """
    #     OUTDATED
    #     Plots RMS distributions for each channel.
    #     """
            
    #     os.makedirs(self.save_dir, exist_ok=True)
    #     fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        
    #     for ch in range(3):
    #         ax = axes[ch]
    #         ax.hist(self.passed['rms'][:, ch], bins=50, range=(0, 10), color="tab:blue", alpha=0.7)
    #         ax.set_xlabel(rf"RMS CH{ch+1} [mV]")
    #         ax.set_ylabel("Counts")
    #         ax.grid(True)

    #     plt.tight_layout()
    #     plt.savefig(os.path.join(self.save_dir, "rms_distribution.pdf"))
    #     plt.close()


    def plot_jitter(self, idx=10):
        """Plots jitter distributions for each channel."""
            
        os.makedirs(self.save_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        sigmas = []
        
        for ch in range(3):
            ax = axes[ch]
            counts, edges, _ = ax.hist(self.passed["w"][:, ch, idx] / self.passed['dvdt_2080'][:, ch] * 1000, bins=50, color="tab:orange", alpha=0.7, label="Data")
            sigma_ufloat, popt, pcov, fit_x, fit_y = self.fit_gauss(counts, edges)
            ax.plot(fit_x, fit_y, linewidth=2, label=fr"Fit: $\sigma={sigma_ufloat}\,$ps", color="red")
            ax.set_xlabel(rf"$N/(dV/dt)$ CH{ch+1} [ps]")
            ax.set_ylabel("Counts")
            ax.grid(True)
            ax.legend()
            sigmas.append(sigma_ufloat)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "jitter_distribution.pdf"))
        plt.close()
        return sigmas

    def plot_cfd(self, cfd_val=0.3):
        """Plots CFD distributions for each channel."""
            
        os.makedirs(self.save_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        cfd_idx = int(10 * cfd_val - 1)
        
        for ch in range(3):
            ax = axes[ch]
            ax.hist(self.passed['cfd'][:, ch, cfd_idx], bins=50, color="tab:orange", alpha=0.7)
            ax.set_xlabel(rf"CFD af {cfd_val} CH{ch+1} [ns]")
            ax.set_ylabel("Counts")
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "cfd_distribution.pdf"))
        plt.close()

    def plot_snr(self, cfd_val=0.3):
        """Plots SNR distributions for each channel."""
            
        os.makedirs(self.save_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        cfd_idx = int(10 * cfd_val - 1)
        
        for ch in range(3):
            ax = axes[ch]
            ax.scatter(self.passed["cfd"][:, ch, cfd_idx], self.passed["pmax"][:, ch] / self.passed['rms'][:, ch], color="tab:orange", alpha=0.7)
            ax.set_xlabel(rf"CFD at {cfd_val} CH{ch+1} [ns]")
            ax.set_ylabel("SNR")
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "snr_distribution.pdf"))
        plt.close()


    def plot_charge(self, transimpedance=4700):
        """Plots charge distributions for each channel."""
            
        os.makedirs(self.save_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        charges = []
        
        for ch in range(3):
            ax = axes[ch]
            charge = 1000 * self.passed['area_new'][:, ch] / transimpedance
            bin_width = 2
            bins = np.arange(0, 150, bin_width)
            counts, edges, _ = ax.hist(charge, bins=bins, color="tab:green", alpha=0.7, label="Data")
            charge, popt, pcov, x_fit, y_fit, amplitude = self.fit_langauss(counts, edges)
            ax.plot(x_fit, y_fit, linewidth=2, label=rf"Langauss Fit: MPV$={charge}\,$fC", color="red")
            # ax.axvline(x=fit_mpv, color="red", linestyle="--", linewidth=2)
            ax.legend()

            ax = axes[ch]
            ax.set_xlabel(rf"Charge CH{ch+1} [fC]")
            ax.set_ylabel("Counts")
            ax.grid(True)

            charges.append(charge)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "charge_distribution.pdf"))
        plt.close()
        return charges


    def plot_wfm_cut_validation(self, channel=0, num_events=50):
        """
        Plots a sample of 'Passed' vs 'Rejected' waveforms side-by-side.
        """
        if not self.passed or not self.rejected:
            raise ValueError("Data cuts not applied. Call apply_cuts() first.")
            
        os.makedirs(self.save_dir, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        
        # Limit how many we draw to avoid massive lag
        draw_passed = min(num_events, self.passed['w'].shape[0])
        draw_rejected = min(num_events, self.rejected['w'].shape[0])

        # Plot Passed Waveforms
        for i in range(draw_passed):
            ax1.plot(self.passed['t'][i, channel, :], 
                     self.passed['w'][i, channel, :], 
                     color='tab:blue', alpha=0.3)
            
        # Plot Rejected Waveforms
        for i in range(draw_rejected):
            ax2.plot(self.rejected['t'][i, channel, :], 
                     self.rejected['w'][i, channel, :], 
                     color='tab:red', alpha=0.3)

        ax1.set_title(f"PASSED Events (Channel {channel+1})")
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('Amplitude (mV)')
        ax1.grid(True)
        ax1.set_xlim(25, 50)

        ax2.set_title(f"REJECTED Events (Channel {channel+1})")
        ax2.set_xlabel('Time (ns)')
        ax2.grid(True)
        ax2.set_xlim(ax1.get_xlim())

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"cut_validation_ch{channel+1}.pdf"))
        plt.close()
    
    
    def plot_amplitude_distribution(self):
        """Plots amplitude histograms and applies Landau fits."""

        if not self.passed:
            raise ValueError("Data cuts not applied. Call apply_cuts() first.")
            
        os.makedirs(self.save_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        
        # We assume 3 channels to plot
        for ch in range(3):
            ax = axes[ch]
            
            # Plot raw vs passed distributions
            raw_pmax = self.raw['pmax'][:, ch]
            passed_pmax = self.passed['pmax'][:, ch]
            
            bin_width = 20
            ax.hist(raw_pmax, bins=100, label="All Data", color="tab:blue", alpha=0.5)
            counts, edges, _ = ax.hist(passed_pmax, bins=np.arange(np.min(passed_pmax), np.max(passed_pmax), bin_width), label="Data (Passed Cuts)", color="tab:green", alpha=0.5)

            mpv, popt, pcov, x_fit, y_fit, amplitude = self.fit_langauss(counts, edges)
            ax.plot(x_fit, y_fit, linewidth=2, label=rf"Langauss Fit: MPV$={mpv}\,$mV", color="red")
                
            ax.set_xlabel(rf"Amplitude CH{ch+1} [mV]")
            ax.set_yscale('log')
            if counts.max() > 0:
                ax.set_ylim(1, counts.max() * 10)
            ax.legend()

        axes[0].set_ylabel("Counts")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "amplitude_dist.pdf"))
        plt.close()


    def analyze_temporal_resolution(self, cfd_val=0.3, fit_func="gaussian"):
        """
        Plots time differences between channels and applies Gaussian fits.
        """
        if not self.passed:
            raise ValueError("Data cuts not applied. Call apply_cuts() first.")

        os.makedirs(self.save_dir, exist_ok=True)

        # Find correct value
        cfd_idx = int(10 * cfd_val - 1)
        
        # Extract the CFD times
        # Shape is [events, channels, thresholds]
        # convert to ps
        t1 = 1000 * self.passed['cfd'][:, 0, cfd_idx]
        t2 = 1000 * self.passed['cfd'][:, 1, cfd_idx]
        t3 = 1000 * self.passed['cfd'][:, 2, cfd_idx]
        
        # Calculate differences (dropping NaNs if any)
        time_diffs = {
            "2-1": (t2 - t1)[~np.isnan(t2 - t1)],
            "3-1": (t3 - t1)[~np.isnan(t3 - t1)],
            "3-2": (t3 - t2)[~np.isnan(t3 - t2)]
        }

        sigmas = []
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        
        for (pair_name, time_vals), ax in zip(time_diffs.items(), axes):
            mean_hist = np.median(time_vals)
            std_hist = np.std(time_vals, ddof=0)
            mad = 8 * np.median(np.abs(time_vals - mean_hist))
            counts, edges, _ = ax.hist(
                time_vals, bins= np.arange(mean_hist - mad, mean_hist + mad, 10),
                label="Data", color="tab:blue", alpha=0.7, edgecolor="black"
            )

            centers = 0.5 * (edges[:-1] + edges[1:])
            initial_guess = [np.sum(counts) * (edges[1] - edges[0]) if len(counts) else 1.0, mean_hist, time_vals.std(ddof=0) or 1.0]
            bounds = ([0.0, -np.inf, 0.0], [np.inf, np.inf, np.inf])
            if fit_func == "student_t":
                initial_guess.append(3.0)
                bounds[0].append(1.0)
                bounds[1].append(np.inf)
            sigma = np.where(counts > 0, np.sqrt(counts), 1.0)

            try:
                popt, pcov = curve_fit(
                    getattr(self, fit_func),
                    centers,
                    counts,
                    p0=initial_guess,
                    sigma = sigma,
                    absolute_sigma=True,
                    bounds=bounds,
                    maxfev=20000,
                )
                fit_x = np.linspace(edges[0], edges[-1], 500)
                ax.plot(fit_x, getattr(self, fit_func)(fit_x, *popt), linewidth=2, label="Gaussian Fit" if fit_func == "gaussian" else "Student t Fit", color="red")
                correlated_params = unc.correlated_values(popt, pcov)

                sigma_ufloat = correlated_params[2]

                if fit_func == "student_t":
                    nu_ufloat = correlated_params[3]
                    if nu_ufloat.nominal_value > 2:
                        sigma_ufloat = sigma_ufloat * sqrt(nu_ufloat / (nu_ufloat - 2))
                    else:
                        # Variance infinite. Fallback to scale
                        pass

                ax.text(
                    0.05, 0.95,
                    rf"$\sigma_{{{pair_name}}}={sigma_ufloat}\,$ps",
                    transform=ax.transAxes,
                    va="top",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )
                sigmas.append(sigma_ufloat)
            except (RuntimeError, ValueError):
                sigmas.append(unc.ufloat(0, 1, tag="ps"))
                ax.text(0.05, 0.95, "Fit failed", transform=ax.transAxes, va="top")

            ax.set_xlabel(rf"ToA$_{{\text{{{pair_name}}}}}$ [ns]")
            ax.legend()

        if all(s.nominal_value > 0 for s in sigmas):
            sigma1_hyp = sigmas[0] / np.sqrt(2)
            sigma1 = sqrt(sigmas[0]**2 + sigmas[1]**2 - sigmas[2]**2) / np.sqrt(2)
            sigma2 = sqrt(sigmas[0]**2 - sigmas[1]**2 + sigmas[2]**2) / np.sqrt(2)
            sigma3 = sqrt(-sigmas[0]**2 + sigmas[1]**2 + sigmas[2]**2) / np.sqrt(2)
            fig.suptitle(rf"$\sigma_{{1, hyp}}={sigma1_hyp}\,$ps, "
                         rf"$\sigma_1={sigma1}\,$ps, "
                         rf"$\sigma_2={sigma2}\,$ps, "
                         rf"$\sigma_3={sigma3}\,$ps")
            print(f"Achieved temporal resolution of sigmahyp={sigma1_hyp}ps, sigma1={sigma1}ps, sigma2={sigma2}ps, sigma3={sigma3}ps")
        else:
            sigma1_hyp = 0
            sigma1 = 0
            sigma2 = 0
            sigma3 = 0
            print("Fit for the temporal resolution failed!")

        axes[0].set_ylabel("Counts")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "temporal_res.pdf"))
        plt.close()
        return sigma1_hyp, sigma1, sigma2, sigma3


    def plot_waveforms(self, event_limit=50,  amplitude_threshold=None):
        """Plots waveforms from the raw dataset where the max amplitude exceeds a specified threshold."""
        if not self.raw:
            self.load_data()
            
        os.makedirs(self.save_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        colors = ['blue', 'red', 'green']
        
        w_data = self.passed['w']
        t_data = self.passed['t']
        
        # Limit loops to available events to prevent out-of-bounds indexing
        max_events = min(event_limit, w_data.shape[0])
        
        for i in range(max_events):
            # for ch in range(3):
            #     # if (amplitude_threshold is not None) & (np.max(w_data[i, ch, :]) < amplitude_threshold):
            #     #     continue
            #     plt.plot(t_data[i, ch, :], w_data[i, ch, :], color=colors[ch], alpha=0.3)
            line = plt.plot(t_data[i, 0, :], w_data[i, 0, :], alpha=0.3)
            plt.plot(t_data[i, 1, :], w_data[i, 1, :], color=line[0].get_color(), alpha=0.3)
            plt.plot(t_data[i, 2, :], w_data[i, 2, :], color=line[0].get_color(), alpha=0.3)
                    
        # Dummy lines for legend
        plt.plot([], [], color='blue', label='Channel 1')
        plt.plot([], [], color='red', label='Channel 2')
        plt.plot([], [], color='green', label='Channel 3')

        plt.xlim(30, 45)

        plt.xlabel('Time (ns)')
        plt.ylabel('Amplitude (mV)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "waveforms.pdf"))
        plt.close()

# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run analysis on ROOT files.")
    
    # Define arguments
    parser.add_argument("file_name", help="Name of the .root file")
    parser.add_argument("voltage", type=int, help="Voltage value (e.g., 225)")
    parser.add_argument("temperature", type=int, choices=[-20, 20], help="Temperature in Celsius")
    parser.add_argument("cut1", type=int, help="Cut value for channel 1")
    parser.add_argument("cut2", type=int, help="Cut value for channel 2")
    parser.add_argument("cut3", type=int, help="Cut value for channel 3")
    
    # Parse the arguments
    args = parser.parse_args()
    # 1. Initialize the analyzer
    analyzer = Analisi(args.file_name, args.voltage, temperature=args.temperature)

    # 2. Load data
    analyzer.load_data()
    
    # Using thresholds: 25 for 170V, 40 for 190V, 70 for 210V
    analyzer.apply_cuts((args.cut1, args.cut2, args.cut3), max_threshold=1200)
    
    # 3. Create Plots
    analyzer.calculate_rms()
    analyzer.plot_amplitude_distribution()
    analyzer.analyze_temporal_resolution(fit_func="gaussian")
    analyzer.plot_wfm_cut_validation(channel=2, num_events=200)
    analyzer.plot_waveforms(amplitude_threshold=1200)
    analyzer.plot_snr()
    analyzer.plot_cfd(cfd_val=0.3)
    charge =analyzer.plot_charge()
    print(charge)
    analyzer.plot_jitter()