"""Name - Vaibhav Gupta
Roll Number - 20IE10041"""

# Loading the packages 
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#Experiment - 1
# Reading from a CSV file
df = pd.read_csv('sales_data.csv')
print (df.head(8))

#Experiment - 2
missing_values_count = df.isnull().sum()

missing_values_table = pd.DataFrame({
    'Column_names': missing_values_count.index,
    'Missing_Values': missing_values_count.values
})

print(missing_values_table)

df = df.fillna(df.mean())

# Print the DataFrame with NaN values replaced by mean
print(df)

#Experiment - 3
df['Date'] =pd.to_datetime(df['Date'], format='%d-%m-%Y')
# Create a line plot
df['Revenue'] =df['Quantity']*df['Price']
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Revenue'],c='red', linewidth = 2)
plt.xticks(rotation ='vertical') 
plt.title('Revenue Trend Over Date')
plt.legend(['Revenue of each day=Quantity_sold*Price_of_Product'],loc='upper center')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.grid(True,alpha=0.9)
plt.show()

#Experiment - 4
total_orders = df['OrderID'].nunique()

# Calculate the total revenue generated from sales
total_revenue = (df['Quantity'] * df['Price']).sum()

# Print the results
print(f'Total Number of Orders: {total_orders}')
print(f'Total Revenue Generated: {total_revenue:.2f}')

#Experiment - 5
average_price = df.groupby('Product').apply(lambda x: (x['Revenue']).sum() / x['Quantity'].sum()).reset_index()
average_price = average_price.rename(columns={0: 'Average_Price'})

print("Average Price of Each Product:")
print(average_price)

# Create a bar plot to visualize the average price of each product
plt.figure(figsize=(5, 6))
plt.bar(average_price['Product'], average_price['Average_Price'],width=0.5, color=['red','green','cyan'])
plt.xlabel('Product')
plt.ylabel('Average Price')
plt.title('Average Price of Each Product')
plt.show()

# Identify and print the top most sold products
top_products = df.groupby('Product')['Quantity'].sum().reset_index()
top_products = top_products.sort_values(by='Quantity', ascending=False)

print("\nTop Most Sold Products:")
print(top_products)