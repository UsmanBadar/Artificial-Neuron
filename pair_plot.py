import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('training_data.csv')

sns.pairplot(data, kind='reg')
plt.show()

