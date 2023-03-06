import numpy as np
import matplotlib.pyplot as plt
y0 = np.arange(0, 11) ** 3
mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180
x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)
x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)
np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
axs[0, 0].plot(np.arange(0, 11), y0)
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')
axs[0, 0].set_title('Graph 1: y = x^3')
axs[0, 0].tick_params(labelsize='x-small')
axs[0, 1].scatter(x1, y1, s=1)
axs[0, 1].set_xlabel('X')
axs[0, 1].set_ylabel('Y')
axs[0, 1].set_title('Graph 2: Scatter plot')
axs[0, 1].tick_params(labelsize='x-small')
axs[1, 0].plot(x2, y2)
axs[1, 0].set_xlabel('X')
axs[1, 0].set_ylabel('Y')
axs[1, 0].set_title('Graph 3: Radioactive decay')
axs[1, 0].tick_params(labelsize='x-small')
axs[1, 1].plot(x3, y31, label='5730 years')
axs[1, 1].plot(x3, y32, label='1600 years')
axs[1, 1].set_xlabel('X')
axs[1, 1].set_ylabel('Y')
axs[1, 1].set_title('Graph 4: Carbon dating')
axs[1, 1].tick_params(labelsize='x-small')
axs[1, 1].legend(fontsize='x-small')
axs[2, 0].hist(student_grades)
axs[2, 0].set_xlabel('Grade')
axs[2, 0].set_ylabel('Frequency')
axs[2, 0].set_title('Graph 5: Distribution of grades')
axs[2, 0].tick_params(labelsize='x-small')
