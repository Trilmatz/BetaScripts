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
    
    def __init__(self, filename, input_path, output_path="plots", tree_name="Analysis"):
        self.filepath = os.path.join(input_path, filename)
        self.save_dir = os.path.join(output_path, filename.split('_')[1], filename.split('_')[2])
        self.tree_name = tree_name
        
        # Load the ROOT file and tree
        self.file = uproot.open(self.filepath)
        self.tree = self.file[self.tree_name]
        
        # Data containers
        self.raw = {}
        self.raw = {}
        self.rejected = {}

        print(f"\n{'='*50}\nInitialized object to analyse measurement stored in {self.filepath}.\n{'='*50}\n")


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

    def fit_gauss(self, counts, edges, initial_guess=None):
        bin_width = edges[1] - edges[0]
        amplitude = np.sum(counts) * bin_width
        centers = 0.5 * (edges[:-1] + edges[1:])
        if initial_guess is None:
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
        branches = self.tree.arrays(["w", "t", "pmax", "negpmax", "event", "rms"])
        
        # Waveforms [events, channels, samples]
        self.raw['w'] = ak.to_numpy(branches["w"]) * 1e3  # in mV
        self.raw['t'] = ak.to_numpy(branches["t"]) * 1e9  # in ns
        
        # Amplitudes [events, channels]
        self.raw['pmax'] = ak.to_numpy(branches["pmax"])
        self.raw['negpmax'] = ak.to_numpy(branches["negpmax"])
        self.raw['rms'] = ak.to_numpy(branches["rms"])

        # Event metadata [events]
        self.raw['event'] = ak.to_numpy(branches["event"])
        
        print(f"Loaded {self.raw['w'].shape[0]} events across {self.raw['w'].shape[1]} channels.")


    def calculate_rms(self, idx=10, save_plot=True):
        """Calculates and plots noise distribution using one point in each waveform."""
        self.raw['noise'] = np.zeros((self.raw['w'].shape[0], 3))
        for channel in range(3):
            self.raw['noise'][:, channel] = self.raw['w'][:, channel, idx]

        os.makedirs(self.save_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        rms_values = []
        
        for ch in range(3):
            ax = axes[ch]
            counts, edges, _ = ax.hist(self.raw['noise'][:, ch], bins=50, range=(-15, 15), color="tab:blue", alpha=0.7, label="Data")
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

    def plot_rms_from_stats(self):
        """
        Plots RMS distributions of the stats file for each channel.
        """
            
        os.makedirs(self.save_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        
        for ch in range(3):
            ax = axes[ch]
            ax.hist(self.raw['rms'][:, ch], bins=50, range=(0, 10), color="tab:blue", alpha=0.7)
            ax.set_xlabel(rf"RMS CH{ch+1} [mV]")
            ax.set_ylabel("Counts")
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "rms_distribution.pdf"))
        plt.close()



    def plot_wfm_cut_validation(self, channel=0, num_events=50):
        """
        Plots a sample of 'raw' vs 'Rejected' waveforms side-by-side.
        """
        if not self.raw or not self.rejected:
            raise ValueError("Data cuts not applied. Call apply_cuts() first.")
            
        os.makedirs(self.save_dir, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        
        # Limit how many we draw to avoid massive lag
        draw_raw = min(num_events, self.raw['w'].shape[0])
        draw_rejected = min(num_events, self.rejected['w'].shape[0])

        # Plot raw Waveforms
        for i in range(draw_raw):
            ax1.plot(self.raw['t'][i, channel, :], 
                     self.raw['w'][i, channel, :], 
                     color='tab:blue', alpha=0.3)
            
        # Plot Rejected Waveforms
        for i in range(draw_rejected):
            ax2.plot(self.rejected['t'][i, channel, :], 
                     self.rejected['w'][i, channel, :], 
                     color='tab:red', alpha=0.3)

        ax1.set_title(f"raw Events (Channel {channel+1})")
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
        """Plots amplitude histograms and applies Langauss fit."""

        if not self.raw:
            raise ValueError("Data cuts not applied. Call apply_cuts() first.")
            
        os.makedirs(self.save_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        
        # We assume 3 channels to plot
        for ch in range(3):
            ax = axes[ch]
            
            # Plot raw vs raw distributions
            raw_pmax = self.raw['pmax'][:, ch]
            raw_pmax = self.raw['pmax'][:, ch]
            
            bin_width = 10
            ax.hist(raw_pmax, bins=100, label="All Data", color="tab:blue", alpha=0.5)
            counts, edges, _ = ax.hist(raw_pmax, bins=np.arange(np.min(raw_pmax), np.max(raw_pmax), bin_width), label="Data (raw Cuts)", color="tab:green", alpha=0.5)

            mpv, popt, pcov, x_fit, y_fit, amplitude = self.fit_langauss(counts, edges)
            ax.plot(x_fit, y_fit, linewidth=2, label=rf"Langauss Fit: MPV$={mpv}\,$mV", color="red")
                
            ax.set_xlabel(rf"Amplitude CH{ch+1} [mV]")
            ax.set_yscale('log')
            if counts.max() > 0:
                ax.set_ylim(0.5, counts.max() * 10)
            ax.legend()

        axes[0].set_ylabel("Counts")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "amplitude_dist.pdf"))
        plt.close()


    def plot_waveforms(self, event_limit=50,  amplitude_threshold=None):
        """Plots waveforms from the raw dataset where the max amplitude exceeds a specified threshold."""
        if not self.raw:
            self.load_data()
            
        os.makedirs(self.save_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        colors = ['blue', 'red', 'green']
        
        w_data = self.raw['w']
        t_data = self.raw['t']
        
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

        # plt.xlim(30, 45)

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
    parser.add_argument("--max_thresholds", nargs=3, type=float, default=[700, 700, 700], help="Maximum threshold values for channels 1, 2, and 3")
    parser.add_argument("--input_path", default="data", help="Path to the input file")
    parser.add_argument("--output_path", default="plots/sensor", help="Path to the output directory")
    
    # Parse the arguments
    args = parser.parse_args()

    if args.file_name == "all":
        rms_values = []
        # If "all" is specified, process all .root files in the input directory
        root_files = [f for f in os.listdir(args.input_path) if f.startswith("stats_") and f.endswith('.root')]
        for file in root_files:
            print(f"\nProcessing file: {file}")
            analyzer = Analisi(file, args.input_path, args.output_path)
            analyzer.load_data()
            rms = analyzer.calculate_rms()
            time = file.split('_')[2]
            rms_values.append((time, rms))

        plt.plot([r[0] for r in rms_values], [r[1][0].n for r in rms_values], label="CH1", marker='o')
        plt.plot([r[0] for r in rms_values], [r[1][1].n for r in rms_values], label="CH2", marker='o')
        plt.plot([r[0] for r in rms_values], [r[1][2].n for r in rms_values], label="CH3", marker='o')
        plt.xticks(rotation=45)
        plt.xlabel("Time")
        plt.ylabel("RMS [mV]")
        plt.title("RMS Values Across All Files")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_path, "rms_across_files.pdf"))
        plt.close()
    
    else:
        # Process a single specified file
        print(f"\nProcessing file: {args.file_name}")

        # 1. Initialize the analyzer
        analyzer = Analisi(args.file_name, args.input_path, args.output_path)

        # 2. Load data
        analyzer.load_data()

        rms = analyzer.calculate_rms()


