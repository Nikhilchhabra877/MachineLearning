#### Developed by : Nikhil ####

import numpy as np
dataset = [11,10,12,14,12,15,14,13,15,102,12,14,17,19,107,10,13,12,14,12,108,12,11,14,13,15,10,15,12,10,14,13,15,10]

# Using Z-score
##################################################
outliers = []
def outliers_Z_Score(data):
    threshold = 3
    mean = np.mean(data)
    std = np.std(data)

    for i in data:
        # outliers = []
        z_score = (i - mean) / std
        if np.abs(z_score) > threshold:
            outliers.append(i)
    return outliers
####################################################
# Using - IQR
out=[]
def outliers_IQR(data):
    dataset = sorted(data)
    Q1,Q3 =np.percentile(dataset,[25,75])
    IQR = Q3 - Q1
    Lower_fence = Q1 - 1.5*IQR
    Higher_fence = Q3 + 1.5*IQR

    for i in dataset :
      if i < Lower_fence or i > Higher_fence:
        out.append(i)
    return out

print("Using Z-score:",outliers_Z_Score(dataset))
print("Using IQR Method:",outliers_IQR(dataset))