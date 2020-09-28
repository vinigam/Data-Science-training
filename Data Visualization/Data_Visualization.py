import pandas as pd 
pd.plotting.register_matplotlib_converters()
import matplotlib as plt 
%matplotlib inline
import seaborn as sns

#Change fig size
plt.figure(figsize=(14,6))
# Add title
plt.title("title")
#LINE CHART (used to continuous data)
	sns.lineplot(data=name_data)
	#subset of data
		# Line chart showing daily global streams of 'Shape of You'
		sns.lineplot(data=spotify_data['Shape of You'], label="Shape of You")

		# Line chart showing daily global streams of 'Despacito'
		sns.lineplot(data=spotify_data['Despacito'], label="Despacito")

		# Add label for horizontal axis
		plt.xlabel("Date")

#Bar char (used to compare data, categorical)

	sns.barplot(x=flight_data.index, y=flight_data['NK'])

#HeatMap(used to see corralations between data)

	sns.heatmap(data=flight_data, annot=True)
	#annot=True - This ensures that the values for each cell appear on the chart. (Leaving this out removes the numbers from each of the cells!)

#Scatter plot (used to see distribution of the data)
	sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])
	# add a regression line
	sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])
	# correlation/distribution between 3 variables
	sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])
	#the line above show in graph the relation and distribution between bmi and charges in xy axis and if is smoker or not
	#regression line in 3 variables graph
	sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data)
	# categorical scatter plot
	sns.swarmplot(x=insurance_data['smoker'],
              y=insurance_data['charges'])

#Histogram(used to compare data, numerical)
sns.distplot(a=iris_data['Petal Length (cm)'], kde=False)
#kde=False is something we'll always provide when creating a histogram, as leaving it out will create a slightly different plot.
	#density plots
	sns.kdeplot(data=iris_data['Petal Length (cm)'], shade=True)
	# 2d kde plot (see 2 variables most  frequenci points)
	sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind="kde")
	#multiple histograms
		# Histograms for each species
		sns.distplot(a=iris_set_data['Petal Length (cm)'], label="Iris-setosa", kde=False)
		sns.distplot(a=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", kde=False)
		sns.distplot(a=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", kde=False)

		# Add title
		plt.title("Histogram of Petal Lengths, by Species")
	#multiple kde plot
		# KDE plots for each species
		sns.kdeplot(data=iris_set_data['Petal Length (cm)'], label="Iris-setosa", shade=True)
		sns.kdeplot(data=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", shade=True)
		sns.kdeplot(data=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", shade=True)

		# Add title
		plt.title("Distribution of Petal Lengths, by Species")
