import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('CrimeRM.csv')

# Preprocess the data
X = data[['Country', 'Individuals Affected']]
y = data['Type Of Cyber Crimes']

# Convert categorical variables to numerical values
X = pd.get_dummies(X, columns=['Country'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply the neural network algorithm
clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=50)
clf.fit(X_train, y_train)

# Evaluate the performance of the neural network algorithm
y_pred_nn = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_nn)

# Apply the decision tree algorithm
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate the performance of the decision tree algorithm
y_pred_dt = model.predict(X_test)
accuracy1 = accuracy_score(y_test, y_pred_dt)
print(f'The accuracy of the decision tree algorithm is {accuracy1:.2f}')

# get the unique country names from the 'country' column
countries = data['Country'].unique()

# print the country names
for Country in countries:
    print(Country)

# ask the user for a Country name
Country = input("Enter the name of the Country: ")
# filter the dataset to get information about cybercrime in a single Country (India) for ransomware and malware
crime_type = ['Ransomware','Malware']
Ransomware_attacks = data[(data['Country'] == Country) & (data['Type Of Cyber Crimes'] == 'Ransomware')]
Malware_attacks = data[(data['Country'] == Country) & (data['Type Of Cyber Crimes'] == 'Malware')]

Total_number_of_attacks = len(Ransomware_attacks) + len(Malware_attacks) 

# get the number of individuals affected and the types of crime
num_individuals_affected_Ransomware = Ransomware_attacks['Individuals Affected'].sum()
num_individuals_affected_Malware = Malware_attacks['Individuals Affected'].sum()

total_num_individuals_affected = num_individuals_affected_Ransomware+  num_individuals_affected_Malware  

# load the population data from a CSV file
populations_df = pd.read_csv('populationF2.csv')

# convert the DataFrame to a dictionary
populations = dict(zip(populations_df['Country'], populations_df['Population']))

# print the population of a given country

if Country in populations:
    print(f"\nThe population of {Country} is {populations[Country]}.\n")

crime_rate = (Total_number_of_attacks/ populations[Country]) * 100000

# print the results

print(f'{num_individuals_affected_Ransomware} individuals were affected by {len(Ransomware_attacks)} ransomware attacks in {Country}.\n')
print(f'{num_individuals_affected_Malware} individuals were affected by {len(Malware_attacks)} malware attacks in {Country}.\n')

print(f'Total {total_num_individuals_affected} individuals were affected by cyber attacks in {Country}.\n')
print(f'Total  cyber attacks in {Country} are {Total_number_of_attacks} .\n')

# Print the accuracy of the neural network algorithm
print(f'The accuracy of the neural network algorithm is {accuracy:.2f}.\n')

# For Comparison between Both the Algorithm

accuracy_list_nn = []
accuracy_list_dt = []

for attack_type in crime_type:
    y_test_attack = y_test[y_test == attack_type]
    y_pred_attack_nn = y_pred_nn[y_test == attack_type]
    y_pred_attack_dt = y_pred_dt[y_test == attack_type]
    accuracy_attack_nn = accuracy_score(y_test_attack, y_pred_attack_nn)
    accuracy_attack_dt = accuracy_score(y_test_attack, y_pred_attack_dt)
    
    accuracy_list_nn.append(accuracy_attack_nn)
    accuracy_list_dt.append(accuracy_attack_dt)

# Plot the graphs side by side

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))

# Plot the first graph: Number of individuals affected by cybercrime
num_individuals_affected = [num_individuals_affected_Ransomware, num_individuals_affected_Malware]
ax1.bar(crime_type, num_individuals_affected)
ax1.set_ylim(0, 10000000)
ax1.set_title(' Cybercrime in {}'.format(Country), fontweight='bold', fontsize=14, color='black')
ax1.set_xlabel('Type of Cybercrime', fontweight='bold', fontsize=14, color='blue')
ax1.set_ylabel('Number of individuals affected (in 10 lacs)', fontweight='bold', fontsize=14, color='blue')
ax1.tick_params(axis='y', pad=20)
ax1.text(-0.1, 1.13, f'Crime Rate: {crime_rate:.4f}', color='red', fontsize=16, fontweight='bold',
         horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes,
         bbox=dict(facecolor='white', edgecolor='black', linewidth=2, boxstyle='round,pad=0.4'))

# Plot the second graph: Comparison
x = range(len(crime_type))
width = 0.35
ax2.bar(x, accuracy_list_nn, width, label='Current Model', color='blue')
ax2.bar([i + width for i in x], accuracy_list_dt, width, label='Existing Model', color='green')
ax2.set_xlabel('Type of Cybercrime',fontweight='bold', fontsize=14, color='blue')
ax2.set_ylabel('Accuracy', fontweight='bold', fontsize=14, color='blue')
ax2.set_title('Comparison',fontweight='bold', fontsize=14, color='black')
ax2.legend()
ax2.set_xticks([i + width/2 for i in x])
ax2.set_xticklabels(crime_type)

# Add a vertical line between the two graphs
fig.subplots_adjust(wspace=0.1)  # Adjust the horizontal space between subplots
line_pos = 0.52  # Position of the vertical line (adjust as needed)
fig.add_artist(plt.Line2D([line_pos, line_pos], [0, 1], color='black', linewidth=2,linestyle='solid'))

# Maximize the figure window
manager = plt.get_current_fig_manager()
manager.window.state('zoomed')

plt.tight_layout()
plt.show()