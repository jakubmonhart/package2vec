import pandas as pd

MIN_COUNT = 2

requirements_path = 'requirements_raw.csv'
output_path = 'requirements_mc{}.csv'.format(MIN_COUNT)

df = pd.read_csv(requirements_path)

# Cleand data
df['dependency'] = df['dependency'].str.lower()
df['dependency'] = df['dependency'].str.replace(r'"', '')
df['dependency'] = df['dependency'].str.replace(r';', '')
df['dependency'] = df['dependency'].str.replace(r' [^\r\n]*', '')
df['dependency'] = df['dependency'].str.replace(r'\[[^\r\n]*', '')

# Save cleaned requirements without aditional filtering
df.to_csv('requirements_cleaned.csv', sep=',', index=False)

# Use only dependencies which occur more then or equal to min_count
counts = df['dependency'].value_counts()
mask = df['dependency'].replace(counts)
df = df.loc[mask.ge(MIN_COUNT)]

# Use only packages with more then 1 dependencies
counts = df['package'].value_counts()
mask = df['package'].replace(counts)
df = df.loc[mask.ge(2)]

df.to_csv(output_path, sep=',', index=False)
