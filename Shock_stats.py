import matplotlib.pyplot as plt
import numpy as np

# Load data
file_path = 'shocks_20250128_180854.dat'  # Update with your actual file path
data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

# Extract y values (sunspot numbers)
y_data = data[:,18]
Density = data[:,9]
Temp = data[:,10]
Magneto_mach = data[:,19]
speed_jump = data[:,8]

binN = 100

bins = np.arange(0,2000,binN)

frequency = np.zeros(20)
summer = np.zeros(20)
for i in range(0,len(y_data)):
    for j in range(0,len(bins)):
        if y_data[i] >= bins[j] and y_data[i] <= bins[j+1]:
            frequency[j] += 1
            #summer[j] += y_data[i]
print(bins)
print(frequency)            
'''
kps = np.arange(1,28)/3
frequency = np.zeros(27)
summer = np.zeros(27)

for i in range(0,len(y_data)):
    for j in range(0,len(kps)):
        if round(x_data[i],1) == round(kps[j],1)*10:
            frequency[j] += 1
            summer[j] += y_data[i]

print(summer)
print(frequency)
average = np.divide(summer,frequency)

print(frequency)
average = np.divide(summer,frequency)
# Convert dates to a numerical format for fitting (e.g., days since start date)


#y_fit = 60 * np.sin((0.003)*x_data + (np.pi - 10/5.5)) + 80
# Plot the data and fitted curve
#print(np.corrcoef(bins, average * -1)[0, 1])
plt.figure(figsize=(10, 6))
plt.scatter(x_data*-1, y_data)
plt.xlabel('DST')
plt.ylabel('SSN')
#plt.scatter(kps, average,color='red')
# Step 4: Add labels and title
#plt.scatter(bins,frequency)
plt.xscale("log")
plt.yscale("log")
#plt.scatter(bins, average , color = "red")
#plt.legend()
plt.show()
'''
binN = 64
#plt.bar(bins, frequency, width=np.diff(bins), align="edge", edgecolor="black", color="blue")
fig, axs = plt.subplots(5, 1, figsize=(8, 20), constrained_layout=True)

axs[0].hist(y_data, bins=binN, density = True, edgecolor="black", color="blue")
axs[0].set_yscale("log")
axs[0].set_xlabel("Bin Range")
axs[0].set_ylabel("Density")
axs[0].set_title("Histogram of Shock Velocity")

# Plot 2: Histogram of Proton Density Ratio
axs[1].hist(Density, bins=binN, density = True, edgecolor="black", color="blue")
axs[1].set_yscale("log")
axs[1].set_xlabel("Bin Range")
axs[1].set_ylabel("Density")
axs[1].set_title("Histogram of Proton Density Ratio")

# Plot 3: Histogram of Proton Temperature
axs[2].hist(Temp, bins=binN, density = True, edgecolor="black", color="blue")
axs[2].set_yscale("log")
axs[2].set_xlabel("Bin Range")
axs[2].set_ylabel("Density")
axs[2].set_title("Histogram of Proton Temperature")

# Plot 4: Histogram of Magnetosonic Mach Number
axs[3].hist(Magneto_mach, bins=binN, density = True, edgecolor="black", color="blue")
axs[3].set_yscale("log")
axs[3].set_xlabel("Bin Range")
axs[3].set_ylabel("Density")
axs[3].set_title("Histogram of Magnetosonic Mach Number")

axs[4].hist(speed_jump, bins=binN, density = True, edgecolor="black", color="blue")
axs[4].set_yscale("log")
axs[4].set_xlabel("Bin Range")
axs[4].set_ylabel("Density")
axs[4].set_title("Histogram of speed jump")
# Show the plots
plt.show()