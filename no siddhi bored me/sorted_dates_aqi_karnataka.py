import pandas as pd

# ===== CONFIG =====
input_file = "cpcb_dly_aq_karnataka-2014.csv"      # put your actual file name here
output_file = "sorted_dataset.csv"   # output file name
date_column = "Sampling Date"        # column name with date
# ===================

# Step 1: Read dataset
df = pd.read_csv(input_file)

# Step 2: Convert the date column into datetime object
# (This handles both dd-mm-yyyy and other mixed formats)
df[date_column] = pd.to_datetime(df[date_column], dayfirst=True, errors='coerce')

# Step 3: Drop rows with invalid or missing dates
df = df.dropna(subset=[date_column])

# Step 4: Sort by the date column
df = df.sort_values(by=date_column)

# Step 5: Save it back to a CSV
df.to_csv(output_file, index=False)

print("✅ Dataset has been sorted successfully and saved as:", output_file)
