import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def rsquared ():
    ss_res = np.sum((bin_centers - x_value) ** 2)
    ss_tot = np.sum((bin_centers - np.mean(bin_centers)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) 
    return r_squared

def graphplot():
    plt.plot(x,T)
    plt.scatter(bin_centers, T_bins, color='red', label="Observed Data")
    plt.scatter(bin_centers, T_bins_rep)
    plt.fill_between(x,T_err1, y_err, color='orange', alpha=0.3, label="Error Bounds")
    plt.legend()
    plt.title('Fitted Fréchet Distribution')
    plt.xlabel('Proton Density Ratio')
    plt.ylabel('Return Period (yrs)')
    plt.show()    

def inverse_pdf(x, target_pdf):
    return stats.genextreme.pdf(x, shape, loc, scale) - target_pdf


file_path = '2023_rainfall_data.csv'  
data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
data = data[:,4]
data = data[data > 0.5]
shape, loc, scale = stats.genextreme.fit(data)
print(scale)
x = np.linspace(min(data), max(data), 1000)
res = np.array([])
for i in range(1,65):
    binN = i
    freq, bins = np.histogram(data, bins=binN, density = True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    time_gap = 49.1671
    nonzero_indices = freq > 0
    freq = freq[nonzero_indices]
    bin_centers = bin_centers[nonzero_indices]
    n = binN/ (9*56.3)
    pdf = stats.genextreme.pdf(x, shape, loc=loc, scale=scale)
    error = stats.genextreme.std(int(shape), loc, scale)
    pdf_err = stats.genextreme.pdf(x, shape, loc=loc, scale=scale +error)
    T = (time_gap/(pdf * len(data))*n)
    T_err1 = ((time_gap/(pdf_err * len(data)))*n)
    T_bins = (time_gap/(freq * len(data)))*n
    x_value = [fsolve(inverse_pdf, x0=bin_center, args=(f,))[0] for bin_center, f in zip(bin_centers, freq)] 
    y_err = (2*T) - T_err1
    res = np.append(res,rsquared ())

plt.ylim(0,1)
plt.xlabel('Number of bins')
plt.ylabel('R-squared value')
plt.scatter(np.arange(1,65,1),res) 
plt.show()

