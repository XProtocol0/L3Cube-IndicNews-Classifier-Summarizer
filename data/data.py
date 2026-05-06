import pandas as pd

# 1. Load your data
# Replace 'your_file.csv' with your actual filename
df = pd.read_csv("/mnt/Shared/SMAI_A3/data/india-news-headlines.csv")

# 2. Define your keywords
# Tip: Use lowercase here; the code below handles case-insensitivity automatically
keywords = [
    # General & Violent Crime
    "crime", "criminal", "murder", "homicide", "manslaughter", "assault",
    "stabbing", "shooting", "kidnapping", "abduction", "rape", "domestic violence",
    
    # Property & Financial Crime
    "robbery", "theft", "burglary", "heist", "arson", "vandalism", "looting",
    "fraud", "embezzlement", "bribery", "extortion", "blackmail", "scam",
    "money laundering", "corruption",
    
    # Organized Crime & Trafficking
    "smuggling", "trafficking", "cartel", "gang", "mafia", "narcotics",
    
    # Law Enforcement & Investigation
    "arrest", "arrested", "suspect", "police", "cops", "detective", "sheriff",
    "investigation", "raid", "warrant", "manhunt", "fugitive", "apprehended",
    
    # Judicial & Penal System
    "indicted", "indictment", "charged", "court", "trial", "lawsuit", "verdict",
    "guilty", "acquitted", "sentenced", "prison", "jail", "inmate", "prosecutor",
    "plea", "bail", "parole",
    
    # Classifications
    "felony", "misdemeanor", "illegal", "illicit", "offense", "violation"
]

# 3. Join keywords into a regex pattern (e.g., "arrested|suspect|robbery")
pattern = '|'.join(keywords)

# 4. Filter the rows
# 'headline' should be the name of the column containing your news titles
filtered_df = df[df['headline_text'].str.contains(pattern, case=False, na=False)]
filtered_df = filtered_df[:5000].reset_index(drop=True)  # Optional: reset index after filtering
# 5. Save the results to a new file
filtered_df.to_csv('/mnt/Shared/SMAI_A3/data/crime_news_filtered.csv', index=False)

print(f"Filtered {len(filtered_df)} rows out of {len(df)}.")