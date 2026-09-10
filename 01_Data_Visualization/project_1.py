import kagglehub
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Tell pandas to show all columns and expand the width limit
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Download dataset to local cache
path = kagglehub.dataset_download("mchirico/montcoalert")

# Point pandas to the CSV inside the cached folder
csv_file = os.path.join(path, "911.csv")

# Load the data into a DataFrame
df = pd.read_csv(csv_file)

# Check the data structure and missing values
df.info()

# Check the first 5 rows of the DataFrame
print("\n--- Head of DataFrame ---")
print(df.head())


#=====================================================================
#QUESTION 01 - 04
#=====================================================================
'''
-->USE THIS METHODS TO ANSWER FIRST FOUR QUESTIONS

1. value_counts() (Returns a Table): 
   This method calculates the frequency of every individual item in a column. 
   Because it needs to show you both the item's name (like a specific zip code) 
   and its corresponding total (the count), it outputs a two-column data 
   structure known as a pandas Series. It answers the question: 
   "What is the count for each value?"

2. nunique() (Returns a Number): 
   In the previous code snippet, the method used was nunique(). 
   The n at the very beginning stands for Number. It asks pandas for the 
   total "Number of Unique values." Since it is only tallying how many 
   distinct categories exist in total (e.g., 148 different emergency titles), 
   it outputs a single integer.

3. unique() (Returns a List): 
   If you drop the 'n' and simply run df['title'].unique(), it will not 
   return a number or a count at all. Instead, it will output a raw 
   array listing the actual text names of every unique emergency code 
   in the dataset.
=====================================================================
'''

'''
=====================================================================
ANSWERS
=====================================================================
'''

# Find the top 5 zip codes for 911 calls and their frequencies
print("\n---------------------------------- Top 5 Zipcodes ------------------------------------------")
print(df['zip'].value_counts().head(5))

# Find the top 5 townships (twp) for 911 calls and their frequencies
print("\n---------------------------------- Top 5 Townships -----------------------------------------")
print(df['twp'].value_counts().head(5))

# Count how many unique title codes exist in the dataset
print("\n----------------------------- Number of Unique Title Codes ---------------------------------")
print(df['title'].nunique())

# Display the raw list of all unique title codes
print("\n------------------------------- List of All Unique Titles ----------------------------------")
print(df['title'].unique())



#=====================================================================
#QUESTION 05
#=====================================================================
'''
--> USE THIS METHOD TO ANSWER THE PART A OF THE 5TH QUESTION


1. The Problem with the Raw Dataset:
   - If you look closely at the columns provided in the raw dataset (lat, lng, desc, 
     zip, title, timeStamp, twp, addr, e), you will notice there is NO standalone 
     column named 'Reason'.
   - If we tried to run .value_counts() directly on the 'title' column without modifying it, 
     it wouldn't give us a clean summary of broad categories. Instead, it would return 
     all 148 distinct, highly specific emergency types separately (e.g., "EMS: BACK PAINS/INJURY", 
     "Fire: GAS-ODOR/LEAK", "EMS: CARDIAC EMERGENCY"), which is too granular to read easily.

2. Discovering Where the Reason is Hidden:
   - By inspecting the text inside the 'title' column, we can see that the information 
     we want is actually bundled together inside the string. Every entry follows a structured 
     format where the broad emergency category comes first, followed by a colon (':'), 
     followed by the specific sub-type.

3. The Solution - Feature Engineering:
   - To solve this, we must perform "Feature Engineering"—meaning we write code to 
     extract, clean, and create a brand-new column called 'Reason' using the existing text.

4. Breakdown of the Extraction Code (.apply, lambda, split, index [0]):
   - df['title']: We target the existing title column containing the full text strings.
   - .apply(...): This pandas method applies a custom operation to every single row 
     in that column independently.
   - lambda title: ... : A mini inline function (an anonymous function) that takes 
     each individual row's text value and temporarily assigns it to the variable 'title'.
   - title.split(':'): This Python string method looks for a colon (':') in the text string 
     and splits it into a list of separate pieces. 
     Example: "EMS: BACK PAINS/INJURY".split(':') turns into the list -> ['EMS', ' BACK PAINS/INJURY']
   - [0]: Because the split operation creates a Python list, we use index 0 to grab 
     the very first element of that list. This isolates just the broad category string ('EMS') 
     and discards everything after the colon.
=====================================================================
'''

df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])

# Display the most common reasons call
print("\n------------------------------- REASON FOR MOST CALLS ----------------------------------")
print(df['Reason'].value_counts().head(3))


'''
--> USE THIS METHOD TO ANSWER THE PART B OF THE QUESTION 05


DETAILED LINE-BY-LINE BREAKDOWN:

1. sns.set_style('whitegrid')
   - What it means: Configures the aesthetic background theme for Seaborn plots.
   - Why we use it: Adds clean, subtle horizontal grid lines across the chart canvas 
     so it's much easier to visually track bar heights against the axis ticks.

2. plt.figure(figsize=(8, 5))
   - What it means: Initializes a new drawing canvas (figure) and sets its physical dimensions.
   - Why we use it: figsize=(8, 5) sets the image to 8 inches wide by 5 inches tall, 
     preventing compressed text or stretched bars.

3. sns.countplot(x='Reason', data=df, hue='Reason', palette='Set2', legend=True)
   - sns.countplot: A specialized function that automatically counts row frequencies 
     and builds a bar chart without requiring manual aggregation.
   - x='Reason': Tells Python which column to place on the horizontal X-axis.
   - data=df: Points Seaborn directly to your main DataFrame.
   - hue='Reason': Maps different colors to each unique category.
   - palette='Set2': Applies a clean, professional soft color theme.
   - legend=True: Renders color mapping cleanly for modern Seaborn versions.

4. plt.title('Total 911 Calls by Emergency Reason')
   - What it means: Assigns a main headline header at the top center of your graph 
     so readers instantly know what data is displayed.

5. plt.xlabel('Reason')
   - What it means: Labels the horizontal axis running along the bottom 
     (showing the categories: EMS, Traffic, Fire).

6. plt.ylabel('Number of Calls')
   - What it means: Labels the vertical axis running up the left side, 
     clarifying that the bar height represents total call frequency.

7. plt.show()
   - What it means: The final execution command that compiles all instructions 
     and pops up the visualization window on your screen. Without this, the graph 
     builds in memory but never displays.
'''

sns.set_style('whitegrid')
plt.figure(figsize=(8, 5))
sns.countplot(x='Reason', data=df, hue='Reason', palette='Set2', legend=True)
plt.title('Total 911 Calls by Emergency Reason')
plt.xlabel('Reason')
plt.ylabel('Number of Calls')
plt.show()


# =====================================================================
# QUESTION 06
# =====================================================================

'''
refer the "Read Me Please!!!!!!!!!" PDF to answer this QUESTION 06
'''


type(df['timeStamp'].iloc[0])
df['timeStamp'] = pd.to_datetime(df['timeStamp'])


df['Hour'] = df['timeStamp'].dt.hour
df['Month'] = df['timeStamp'].dt.month
df['Day of Week'] = df['timeStamp'].dt.dayofweek

# Map integer days to text names using your dictionary
dmap = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)


# =====================================================================
# QUESTION 6A: 911 CALLS BY DAY OF WEEK (COLORED BY REASON)
# =====================================================================
sns.set_style('whitegrid')
plt.figure(figsize=(10, 6))
sns.countplot(x='Day of Week', data=df, hue='Reason', palette='viridis')
plt.title('Total 911 Calls by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Calls')
plt.legend(title='Reason', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# =====================================================================
# QUESTION 6B: 911 CALLS BY MONTH (COLORED BY REASON)
# =====================================================================
sns.set_style('whitegrid')
plt.figure(figsize=(12, 6))
sns.countplot(x='Month', data=df, hue='Reason', palette='viridis')
plt.title('Total 911 Calls by Month')
plt.xlabel('Month')
plt.ylabel('Number of Calls')
plt.legend(title='Reason', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# =====================================================================
# QUESTION 07
# =====================================================================
'''
--> QUESTION 07 A
Explanation:
- Instead of counting categories using bar charts, we want to see how the 
  total volume of calls trends across different months of the year.
- df.groupby('Month').count(): Groups all rows by their month number and 
  counts occurrences across every column.
- byMonth['twp'].plot(): Takes any column (like township 'twp') from the 
  aggregated table and renders a standard line plot.
'''
byMonth = df.groupby('Month').count()

plt.figure(figsize=(10, 6))
byMonth['twp'].plot()
plt.title('Monthly 911 Calls Trend')
plt.xlabel('Month')
plt.ylabel('Number of Calls')
plt.show()


'''
--> QUESTION 07B
Explanation:
- sns.lmplot(): A Seaborn function that creates a scatter plot combined with 
  a linear regression line (best-fit line) and a confidence interval band.
- data=byMonth.reset_index(): Because groupby turns 'Month' into the index 
  of our DataFrame, reset_index() pushes it back into a standard column so 
  Seaborn can read it on the X-axis.
'''
sns.lmplot(x='Month', y='twp', data=byMonth.reset_index())
plt.title('Linear Fit of Calls per Month')
plt.show()



'''
QUESTION 07C
Explanation:
- First, we engineer a new 'Date' column by stripping away the hours, minutes, 
  and seconds from the 'timeStamp' using the .date() method.
- Then, we group the entire DataFrame by this new 'Date' column, count the calls 
  for each specific calendar day, and plot the result as a continuous time-series line.
'''
# Create a Date column
df['Date'] = df['timeStamp'].apply(lambda t: t.date())

# Group by Date and plot call volumes over time
plt.figure(figsize=(12, 6))
df.groupby('Date').count()['twp'].plot()
plt.title('Daily 911 Calls Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Calls')
plt.tight_layout()
plt.show()


# =====================================================================
# QUESTION 08: CREATING A HEATMAP (DAY OF WEEK VS. HOUR)
# =====================================================================
'''Refer to the "Read Me Please!!!!!!!!!" PDF'''
# Step 1: Create a matrix table grouping by Day of Week and Hour
dayHour = df.groupby(by=['Day of Week', 'Hour']).count()['Reason'].unstack()

# Step 2: Render the heatmap using Seaborn
plt.figure(figsize=(12, 6))
sns.heatmap(dayHour, cmap='viridis')

plt.title('HeatMap of 911 Calls: Day of Week vs. Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Day of Week')
plt.show()


# =====================================================================
# QUESTION 09: CREATING A CLUSTERMAP (DAY OF WEEK VS. HOUR)
# =====================================================================
'''
Explanation:
- sns.clustermap(): Similar to sns.heatmap(), but it calculates hierarchical 
  clustering to automatically reorder and group rows (days) and columns (hours) 
  that share similar patterns or trends together.
'''
sns.clustermap(dayHour, cmap='viridis')
plt.show()