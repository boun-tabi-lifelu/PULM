import sqlite3
import random
import pandas as pd
import numpy as np

# Configuration
DB_PATH = "/cta/share/users/uniprot/uniref/uniref50_2026_02/uniref50_representatives.db"
SOURCE_TABLE = "plm_train"
TARGET_TABLE = "tokenizer_train"
TARGET_ROWS = 2000000

def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("Finding table boundaries...")
    cursor.execute(f"SELECT MAX(rowid) FROM {SOURCE_TABLE}")
    max_rowid = cursor.fetchone()[0]
    
    # We pull 2.5M rows initially to account for any rowid gaps 
    # and rows dropped during the ambiguous amino acid filtering step.
    oversample_size = int(TARGET_ROWS * 1.25) 
    print(f"Generating {oversample_size} random target IDs...")
    random_ids = random.sample(range(1, max_rowid + 1), oversample_size)
    
    # Fetch in chunks to keep memory predictable and avoid giant SQL strings
    chunk_size = 100000
    dfs = []
    
    print("Fetching random samples from SQLite...")
    for i in range(0, len(random_ids), chunk_size):
        chunk_ids = random_ids[i:i+chunk_size]
        placeholders = ','.join('?' for _ in chunk_ids)
        query = f"SELECT entry_id, sequence FROM {SOURCE_TABLE} WHERE rowid IN ({placeholders})"
        
        df_chunk = pd.read_sql_query(query, conn, params=chunk_ids)
        dfs.append(df_chunk)
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} initial samples. Applying character filters...")
    
    # 1. Condition Filters: Count of X, B, U, Z must be <= 1
    mask = (
        (df['sequence'].str.count('X') <= 1) &
        (df['sequence'].str.count('B') <= 1) &
        (df['sequence'].str.count('U') <= 1) &
        (df['sequence'].str.count('Z') <= 1)
    )
    df = df[mask].copy()
    print(f"Rows remaining after character filtering: {len(df)}")
    
    if len(df) < TARGET_ROWS:
        raise ValueError(f"Only {len(df)} rows passed filters. Increase oversample_size.")
        
    # Trim down to exactly 2,000,000 rows
    df = df.sample(n=TARGET_ROWS, random_state=42).reset_index(drop=True)
    
    # 2. Sequential Window Slicing for lengths > 3000
    print("Slicing long sequences into random 3000-length windows...")
    def get_consecutive_chunk(seq):
        seq_len = len(seq)
        if seq_len > 3000:
            # Pick a random valid starting position
            start_idx = np.random.randint(0, seq_len - 3000 + 1)
            return seq[start_idx:start_idx + 3000]
        return seq

    df['sequence'] = df['sequence'].apply(get_consecutive_chunk)
    
    # 3. Save back to the database
    print(f"Saving final dataframe to table '{TARGET_TABLE}'...")
    df.to_sql(TARGET_TABLE, conn, if_exists='replace', index=False)
    
    # Optional: Index entry_id if you plan to query individual sequences later
    print("Creating index on entry_id...")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_tokenizer_entry ON {TARGET_TABLE}(entry_id)")
    
    conn.close()
    print("Process complete!")

if __name__ == "__main__":
    main()