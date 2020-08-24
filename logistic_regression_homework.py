import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

## INPUT DATA
heart_rates = np.array([50, 70, 90]).reshape(-1, 1)
logits = [-1.09861, 0,  1.38629]

## Create linear regressor
LR = LinearRegression()
LR.fit(heart_rates, logits)
print("Weights:", LR.coef_)
print("BIAS:", LR.intercept_)

## PREDICT for HR=60

## Example - plot the data / best fit line
## How would we plot the line transformed back into probability space (i.e. the sigmoid one)?
plt.scatter(heart_rates, logits)
x = np.linspace([0], [150])  # create a list of evenly spaced values between 0 and 150 - your x values
y = LR.predict(x)  # this gives you the logits (would need to run this through a sigmoid to get the probability)
plt.plot(x, y)
#plt.show()


# Same for logistic regression, but note it expects the original data (not the logit version)
all_heart_rates = [50, 50, 50, 50, 70, 70, 90, 90, 90, 90, 90]
heart_attacks = [True, False, False, False, False, True, True, True, False, True, True]
LOR = LogisticRegression(random_state=0, solver='lbfgs')

LOR.fit(all_heart_rates,  heart_attacks)
# this will return an array: prob_of_heart_attack, 1-prob_of_heart_attack (or the other
# way around, depending on what binary representation of heart attack you used)
LOR.predict_proba(all_heart_rates)
