import pandas as pd

# 1. Load dataset
df = pd.read_csv('/mnt/Shared/SMAI_A3/data/Data.csv')

# 2. Clean column
df['headline_category'] = df['headline_category'].astype(str).str.strip()

# 3. Target categories
target_categories = [
    "Business", "Crime", "Entertainment",
    "Politics", "Sports", "Technology", "Finance"
]

# 4. Filter
filtered_df = df[df['headline_category'].isin(target_categories)].copy()

# 5. Target count
target_count = 1000

# 6. Balance manually (NO groupby.apply)
balanced_list = []

for category in target_categories:
    subset = filtered_df[filtered_df['headline_category'] == category]
    
    if len(subset) == 0:
        continue  # skip missing categories
    
    sampled = subset.sample(
        n=min(len(subset), target_count),
        random_state=42
    )
    
    balanced_list.append(sampled)

# Combine everything
balanced_df = pd.concat(balanced_list, ignore_index=True)

# 7. Shuffle
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 8. Save
balanced_df.to_csv('/mnt/Shared/SMAI_A3/data/filtered_data1.csv', index=False)

# 9. Verify
print("\n--- Original Counts ---")
print(filtered_df['headline_category'].value_counts())

print("\n--- Balanced Counts ---")
print(balanced_df['headline_category'].value_counts())

print("\nColumns:")
print(balanced_df.columns)