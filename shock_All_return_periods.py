import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

file_path = 'shocks_20250128_180854.dat'  # Update with your actual file path
data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
# Generate some synthetic data (replace with your own data)

data = data[:,8]

# Sort data in descending order (largest event first)
data_sorted = np.sort(data)[::-1]
n = len(data)

# Compute empirical return periods
rank = np.arange(1, n + 1)
return_periods = (n + 1) / rank  # Weibull plotting position formula

# Fit distributions (Weibull, Gumbel, Fréchet/GEV)
weibull_params = stats.weibull_min.fit(data)
gumbel_params = stats.gumbel_r.fit(data)
frechet_params = stats.genextreme.fit(data)

# Generate return periods for plotting
T_values = np.logspace(np.log10(1), np.log10(1000), 1000)  # Return periods from 1 to 1000

# Compute return levels using the fitted distributions
weibull_return_levels = stats.weibull_min.ppf(1 - 1/T_values, *weibull_params)
gumbel_return_levels = stats.gumbel_r.ppf(1 - 1/T_values, *gumbel_params)
frechet_return_levels = stats.genextreme.ppf(1 - 1/T_values, *frechet_params)

# Plot observed vs fitted return periods
"""

plt.figure(figsize=(8, 6))
plt.scatter(return_periods, data_sorted, label="Observed Data", color='black', marker="o")
#plt.plot(T_values, weibull_return_levels, label="Weibull Fit", linestyle="--", color="red")
plt.plot(T_values, gumbel_return_levels, label="Gumbel Fit", linestyle="--", color="blue")
plt.plot(T_values, frechet_return_levels, label="Fréchet (GEV) Fit", linestyle="--", color="green")

# Formatting
plt.xscale("log")  # Log scale for return periods
plt.yscale("linear")  # Linear scale for values
plt.xlabel("Return Period (Years)")
plt.ylabel("Return Level")
plt.title("Return Period Plot")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
"""


fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

"""
axs.scatter(return_periods, data_sorted, label="Observed Data", color='black', marker="o")
axs.plot(T_values, gumbel_return_levels, label="Gumbel Fit", linestyle="--", color="blue")
axs.set_title('Gumbel Fit Shock speed')
axs.set_xscale("log")
axs.set_yscale("linear")
axs.set_xlabel("Return Period (Years)")
axs.set_ylabel("Shock Speed (km/s)")


# Plot 2: Histogram of Proton Density Ratio
axs.scatter(return_periods, data_sorted, label="Observed Data", color='black', marker="o")
axs.plot(T_values, frechet_return_levels, label="Fréchet (GEV) Fit", linestyle="--", color="green")
axs.set_title('Frechet Fit Shock speed')
axs.set_xscale("log")
axs.set_yscale("linear")
axs.set_xlabel("Return Period (Years)")
axs.set_ylabel("Shock Speed (km/s)")
"""
axs.scatter(return_periods, data_sorted, label="Observed Data", color='black', marker="o")
axs.plot(T_values, weibull_return_levels, label="Fréchet (GEV) Fit", linestyle="--", color="red")
axs.set_title('Weibull Fit Shock speed')
axs.set_xscale("log")
axs.set_yscale("linear")
axs.set_xlabel("Return Period (Years)")
axs.set_ylabel("Shock Speed (km/s)")
plt.show()
