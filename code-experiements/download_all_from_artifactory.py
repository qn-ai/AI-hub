# --- 1. Summary Statistics ---
print("Summary statistics for 'current_age':")
print(df['current_age'].describe(), "\n")

# --- 2. Value Counts (Frequency Table) ---
print("Age frequency distribution:")
print(df['current_age'].value_counts().sort_index(), "\n")

# --- 3. Visual Distribution (Histogram) ---
plt.figure(figsize=(8, 5))
plt.hist(df['current_age'], bins=10, edgecolor='black')
plt.title('Distribution of Current Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
