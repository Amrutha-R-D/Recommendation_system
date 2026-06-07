import pandas as pd
import numpy as np
import os

# ──────────────────────────────────────────────────────────────────────
# SMART DYNAMIC PATH RESOLUTION (No changes needed for your machine!)
# ──────────────────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.basename(CURRENT_DIR) == "data":
    BASE_DIR = CURRENT_DIR
else:
    BASE_DIR = os.path.join(CURRENT_DIR, "data")

print(f"[*] Pipeline executing. Target data directory: {BASE_DIR}")

# ==========================================
# 1. PARSE MOVIE TITLES
# ==========================================
def process_movie_titles(filepath):
    print("Processing movie titles...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing file: {filepath}")
        
    df_movies = pd.read_csv(filepath, encoding="ISO-8859-1", header=None, 
                            names=['Movie_ID', 'Year', 'Title'], engine='python', on_bad_lines='skip')
    df_movies['Movie_ID'] = df_movies['Movie_ID'].astype(np.int32)
    
    output_path = os.path.join(BASE_DIR, "df_movies.parquet")
    df_movies.to_parquet(output_path, index=False)
    print(f"Saved {output_path}")

# ==========================================
# 2. PARSE THE PROBE SET (Validation Indices)
# ==========================================
def parse_probe(filepath):
    print("Parsing probe file...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing file: {filepath}")
        
    probe_list = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.endswith(':'):
                current_movie_id = int(line[:-1])
            else:
                probe_list.append((current_movie_id, int(line)))
    df_probe = pd.DataFrame(probe_list, columns=['Movie_ID', 'User_ID'])
    df_probe['Movie_ID'] = df_probe['Movie_ID'].astype(np.int32)
    df_probe['User_ID'] = df_probe['User_ID'].astype(np.int32)
    return df_probe

# ==========================================
# 3. PARSE COMBINED DATA & SPLIT TRAIN/VAL
# ==========================================
def process_combined_data(file_paths, df_probe):
    print("Processing combined data files (this will take a few minutes)...")
    
    temp_train_path = os.path.join(BASE_DIR, "temp_train.csv")
    temp_val_path = os.path.join(BASE_DIR, "temp_val.csv")
    
    train_file = open(temp_train_path, "w")
    val_file = open(temp_val_path, "w")
    
    headers = "Movie_ID,User_ID,Rating,Date\n"
    train_file.write(headers)
    val_file.write(headers)
    
    probe_set = set(zip(df_probe['Movie_ID'], df_probe['User_ID']))
    
    for path in file_paths:
        if not os.path.exists(path):
            print(f"[!] Skipping {path} (File not found)")
            continue
        print(f"Reading {path}...")
        with open(path, 'r') as f:
            current_movie_id = None
            for line in f:
                line = line.strip()
                if line.endswith(':'):
                    current_movie_id = line[:-1]
                else:
                    user_id, rating, date = line.split(',')
                    row_str = f"{current_movie_id},{user_id},{rating},{date}\n"
                    
                    if (int(current_movie_id), int(user_id)) in probe_set:
                        val_file.write(row_str)
                    else:
                        train_file.write(row_str)
                        
    train_file.close()
    val_file.close()
    
    print("Compressing data into Parquet format...")
    for name in ["train", "val"]:
        temp_path = temp_train_path if name == "train" else temp_val_path
        if os.path.exists(temp_path):
            df = pd.read_csv(temp_path)
            df['Movie_ID'] = df['Movie_ID'].astype(np.int32)
            df['User_ID'] = df['User_ID'].astype(np.int32)
            df['Rating'] = df['Rating'].astype(np.int8)
            df['Date'] = pd.to_datetime(df['Date'])
            
            output_parquet = os.path.join(BASE_DIR, f"df_{name}.parquet")
            df.to_parquet(output_parquet, index=False)
            os.remove(temp_path) 
        
    print(f"Saved df_train.parquet and df_val.parquet inside {BASE_DIR}")

# ==========================================
# 4. PARSE QUALIFYING SET (Unlabeled Test)
# ==========================================
def process_qualifying(filepath):
    print("Processing qualifying file...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing file: {filepath}")
        
    qual_list = []
    with open(filepath, 'r') as f:
        current_movie_id = None
        for line in f:
            line = line.strip()
            if line.endswith(':'):
                current_movie_id = int(line[:-1])
            else:
                user_id, date = line.split(',')
                qual_list.append((current_movie_id, int(user_id), date))
                
    df_qual = pd.DataFrame(qual_list, columns=['Movie_ID', 'User_ID', 'Date'])
    df_qual['Movie_ID'] = df_qual['Movie_ID'].astype(np.int32)
    df_qual['User_ID'] = df_qual['User_ID'].astype(np.int32)
    df_qual['Date'] = pd.to_datetime(df_qual['Date'])
    
    output_path = os.path.join(BASE_DIR, "df_test_unlabeled.parquet")
    df_qual.to_parquet(output_path, index=False)
    print(f"Saved {output_path}")

# ==========================================
# EXECUTION ORCHESTRATION
# ==========================================
if __name__ == "__main__":
    movie_titles_file = os.path.join(BASE_DIR, "movie_titles.csv")
    probe_file = os.path.join(BASE_DIR, "probe.txt")
    qualifying_file = os.path.join(BASE_DIR, "qualifying.txt")
    
    combined_files = [
        os.path.join(BASE_DIR, "combined_data_1.txt"),
        os.path.join(BASE_DIR, "combined_data_2.txt"),
        os.path.join(BASE_DIR, "combined_data_3.txt"),
        os.path.join(BASE_DIR, "combined_data_4.txt")
    ]
    
    process_movie_titles(movie_titles_file)
    df_probe = parse_probe(probe_file)
    process_combined_data(combined_files, df_probe)
    process_qualifying(qualifying_file)
    
    print("\n--- Pipeline Complete! Ready for Model Training ---")