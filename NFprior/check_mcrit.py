import numpy as np

save_data_filename = "./models/uniform/bns/radio_mcrit/training_data.npz"
data = np.load(save_data_filename)
mcrit_ratio = np.array(data["mcrit_ratio"])
chi_1 = data["chi_1"]
chi_2 = data["chi_2"]
mtov = np.array(data["mtov"])
mcrit = mcrit_ratio * mtov

chi = np.concatenate((chi_1, chi_2))
max_chi = np.max(chi)
max_chi = np.round(max_chi, 2)

# Make a histogram of the mcrit ratio
import matplotlib.pyplot as plt
plt.hist(mcrit_ratio, bins=50, density=True)
plt.xlabel("Mcrit / MTOV")
plt.ylabel("Density")
plt.title("Histogram of Mcrit / MTOV ratios")
plt.grid()
plt.savefig(f"./figures/mcrit_ratio_histogram_{max_chi:.2f}.pdf", bbox_inches="tight")
plt.close()
 
# Now of the chi
plt.figure()
plt.hist(chi, bins=50, density=True)
plt.xlabel("Chi")
plt.ylabel("Density")
plt.title("Histogram of Chi values")
plt.grid()
plt.savefig("./figures/chi_histogram.pdf", bbox_inches="tight")
print(mcrit_ratio)
plt.close()

# Now make a histogram of Mcrit and Mtov with legend:
plt.figure()
plt.hist(mcrit, bins=50, density=True, alpha=0.5, label="Mcrit", color='blue')
plt.hist(mtov, bins=50, density=True, alpha=0.5, label="MTOV", color='orange')
plt.xlabel("Mass (M_sun)")
plt.ylabel("Density")
plt.title("Histogram of Mcrit and MTOV")
plt.legend()
plt.grid()
plt.savefig(f"./figures/mcrit_mtov_histogram_{max_chi:.2f}.pdf", bbox_inches="tight")
plt.close()