import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

df = pd.read_csv("../data/covid_data.csv")
df.head()

df.shape
df.columns
df.info()
df.describe()

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
print(df.columns.tolist())

required_cols = [
    'location',
    'date',
    'total_cases',
    'new_cases',
    'total_deaths',
    'new_deaths'
]

# Vaccination-related possible columns
vaccine_options = [
    'people_vaccinated',
    'total_vaccinations',
    'people_fully_vaccinated'
]

available_cols = df.columns.tolist()

# Keep only columns that exist
final_cols = [col for col in required_cols if col in available_cols]

# Add first available vaccination column
for vcol in vaccine_options:
    if vcol in available_cols:
        final_cols.append(vcol)
        break

df = df[final_cols]

df.head()

df = df.sort_values(['location', 'date'])

df = df.groupby('location').apply(lambda x: x.ffill()).reset_index(drop=True)
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

df['new_cases_7day_avg'] = (
    df.groupby('location')['new_cases']
    .transform(lambda x: x.rolling(7).mean())
)
df['death_rate'] = np.where(
    df['total_cases'] > 0,
    df['total_deaths'] / df['total_cases'],
    0
)

global_cases = df.groupby('date')['new_cases'].sum().reset_index()

sns.lineplot(data=global_cases, x='date', y='new_cases')
plt.title("Global Daily New COVID Cases")
plt.xlabel("Date")
plt.ylabel("New Cases")
plt.show()

global_avg = df.groupby('date')['new_cases_7day_avg'].sum().reset_index()

sns.lineplot(data=global_avg, x='date', y='new_cases_7day_avg')
plt.title("Global 7-Day Average of New Cases")
plt.xlabel("Date")
plt.ylabel("7-Day Avg Cases")
plt.show()

countries = ['India', 'United States', 'Brazil']

country_df = df[df['location'].isin(countries)]

sns.lineplot(
    data=country_df,
    x='date',
    y='new_cases',
    hue='location'
)

plt.title("Daily COVID Cases Comparison")
plt.show()


sns.lineplot(
    data=country_df,
    x='date',
    y='death_rate',
    hue='location'
)

plt.title("Death Rate Comparison")
plt.show()

peak_day = global_cases.loc[global_cases['new_cases'].idxmax()]

print("Peak Global Case Day:")
print(peak_day)