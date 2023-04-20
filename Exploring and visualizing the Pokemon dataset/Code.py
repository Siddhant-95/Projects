
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the dataset
df = pd.read_csv('pokemon.csv')
df
df.info()

# Dropping duplicates and missing values
df.drop_duplicates(inplace = True)
df.dropna(inplace = True)

df.info()

# Number of Pokemon per Generation
df.generation.value_counts().plot(kind = 'bar')

# Names of Legendary Pokemon
df['name'][(df['is_legendary'] == 1)]

# Plotting the types of Pokemon
df['type1'].value_counts().plot(kind = 'bar')

# Isolating important features
df1 = df[['hp','attack','defense','speed','sp_attack','sp_defense']]

# Scatter plot among the important features
pd.plotting.scatter_matrix(df1)
plt.show()

for i in df1.columns:
    df1[i].plot(kind = 'kde')
    plt.xlabel(i)
    plt.show()

# Boxplot to check outliers
df1.boxplot(column = ['hp','attack','defense','speed','sp_attack','sp_defense'])
plt.show()

# Attack vs Defense scatter plot
plt.scatter(df1.attack, df1.defense)
plt.xlabel('Attack')
plt.ylabel('Defense')
plt.show()


# Heatmap to check correlation between the features
corr_matrix = df1.corr()
sns.heatmap(corr_matrix, annot =True)
plt.show()


# Pokemon which also have a secondary type
plt.figure(figsize = (20,5))
type1 = df.type1
type2 = df.type2
both_types = type1 + '-' + type2
both_types.value_counts().plot(kind = 'bar')
plt.show()


# Statistical Analysis
mean_attack = df1.attack.mean()
median_attack = df1.attack.median()
std_dev_attack = df1.attack.std()
mean_defense = df1.defense.mean()
median_defense = df1.defense.median()
std_dev_defense = df1.defense.std()
mean_hp = df1.hp.mean()
median_hp = df1.hp.median()
std_dev_hp = df1.hp.std()
mean_speed = df1.speed.mean()
median_speed = df1.speed.median()
std_dev_speed = df1.speed.std()
mean_sp_attack = df1.sp_attack.mean()
median_sp_attack = df1.sp_attack.median()
std_dev_sp_attack = df1.sp_attack.std()
mean_sp_defense = df1.sp_defense.mean()
median_sp_defense = df1.sp_defense.median()
std_dev_sp_defense = df1.sp_defense.std()


print('Attack Power Mean:',mean_attack)
print('Attack Median:',median_attack)
print('Attack Power Standard Dev:',std_dev_attack,'\n')
print('Defense Power Mean:',mean_defense)
print('Defense Power Median:',median_defense)
print('Defense Power Standard Dev:',std_dev_defense,'\n')
print('HP Mean:',mean_hp)
print('HP Median:',median_hp)
print('HP Standard Dev:',std_dev_hp,'\n')
print('Speed Mean:',mean_speed)
print('Speed Median:',median_speed)
print('Speed Standard Dev:',std_dev_speed,'\n')
print('Attack Speed Mean:',mean_sp_attack)
print('Attack Speed Median:',median_sp_attack)
print('Attack Speed Standard Dev:',std_dev_sp_attack,'\n')
print('Defense Speed Mean:',mean_sp_defense)
print('Defense Speed Median:',mean_sp_defense)
print('Defense Speed Standard Dev:',std_dev_sp_defense)
