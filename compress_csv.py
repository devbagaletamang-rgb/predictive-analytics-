"""
CSV Compression Script
Converts CSV files from 1.02 MB to ~500 KB using multiple strategies
"""

import pandas as pd
import os

def get_file_size_mb(filepath):
    """Get file size in MB"""
    return os.path.getsize(filepath) / (1024 * 1024)

def compress_csv_remove_columns(input_file, output_file, columns_to_keep):
    """
    Strategy 1: Remove unnecessary columns
    """
    print(f"Original file size: {get_file_size_mb(input_file):.2f} MB")
    
    df = pd.read_csv(input_file)
    print(f"Shape before: {df.shape}")
    
    # Keep only specified columns
    df = df[columns_to_keep]
    df.to_csv(output_file, index=False)
    
    print(f"Compressed file size: {get_file_size_mb(output_file):.2f} MB")
    print(f"Shape after: {df.shape}")

def compress_csv_sample(input_file, output_file, fraction=0.5):
    """
    Strategy 2: Downsample rows (keep 50% randomly)
    """
    print(f"Original file size: {get_file_size_mb(input_file):.2f} MB")
    
    df = pd.read_csv(input_file)
    print(f"Shape before: {df.shape}")
    
    # Sample 50% of rows
    df = df.sample(frac=fraction, random_state=42)
    df.to_csv(output_file, index=False)
    
    print(f"Compressed file size: {get_file_size_mb(output_file):.2f} MB")
    print(f"Shape after: {df.shape}")

def compress_csv_duplicates(input_file, output_file):
    """
    Strategy 3: Remove duplicate rows
    """
    print(f"Original file size: {get_file_size_mb(input_file):.2f} MB")
    
    df = pd.read_csv(input_file)
    print(f"Shape before: {df.shape}")
    
    # Remove duplicates
    df = df.drop_duplicates()
    df.to_csv(output_file, index=False)
    
    print(f"Compressed file size: {get_file_size_mb(output_file):.2f} MB")
    print(f"Shape after: {df.shape}")

def compress_csv_gzip(input_file, output_file):
    """
    Strategy 4: Compress with Gzip (keeps all data)
    Best compression ratio
    """
    print(f"Original file size: {get_file_size_mb(input_file):.2f} MB")
    
    df = pd.read_csv(input_file)
    
    # Save as gzip compressed CSV
    df.to_csv(output_file, index=False, compression='gzip')
    
    print(f"Compressed file size: {get_file_size_mb(output_file):.2f} MB")

def compress_csv_precision(input_file, output_file, decimal_places=2):
    """
    Strategy 5: Reduce decimal precision for numeric columns
    """
    print(f"Original file size: {get_file_size_mb(input_file):.2f} MB")
    
    df = pd.read_csv(input_file)
    print(f"Shape before: {df.shape}")
    
    # Round numeric columns
    numeric_cols = df.select_dtypes(include=['float64']).columns
    df[numeric_cols] = df[numeric_cols].round(decimal_places)
    
    df.to_csv(output_file, index=False)
    
    print(f"Compressed file size: {get_file_size_mb(output_file):.2f} MB")
    print(f"Shape after: {df.shape}")

def compress_csv_datatypes(input_file, output_file):
    """
    Strategy 6: Optimize data types (int64→int32, float64→float32)
    """
    print(f"Original file size: {get_file_size_mb(input_file):.2f} MB")
    
    df = pd.read_csv(input_file)
    print(f"Shape before: {df.shape}")
    print(f"Dtypes before:\n{df.dtypes}\n")
    
    # Optimize data types
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    
    df.to_csv(output_file, index=False)
    
    print(f"Compressed file size: {get_file_size_mb(output_file):.2f} MB")
    print(f"Shape after: {df.shape}")
    print(f"Dtypes after:\n{df.dtypes}\n")

def compress_csv_parquet(input_file, output_file):
    """
    Strategy 7: Convert to Parquet format (50-80% smaller)
    """
    print(f"Original CSV size: {get_file_size_mb(input_file):.2f} MB")
    
    df = pd.read_csv(input_file)
    
    # Save as Parquet
    df.to_parquet(output_file, index=False, compression='snappy')
    
    print(f"Parquet file size: {get_file_size_mb(output_file):.2f} MB")

def compress_csv_combined(input_file, output_file):
    """
    Strategy 8: Combined approach (remove duplicates + reduce precision + optimize types)
    RECOMMENDED: Best balance between compression and data integrity
    """
    print(f"Original file size: {get_file_size_mb(input_file):.2f} MB")
    
    df = pd.read_csv(input_file)
    print(f"Shape before: {df.shape}")
    
    # Step 1: Remove duplicates
    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape}")
    
    # Step 2: Reduce decimal precision
    numeric_cols = df.select_dtypes(include=['float64']).columns
    df[numeric_cols] = df[numeric_cols].round(2)
    
    # Step 3: Optimize data types
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    
    df.to_csv(output_file, index=False)
    
    print(f"Compressed file size: {get_file_size_mb(output_file):.2f} MB")
    print(f"Shape after: {df.shape}")
    print("✓ Successfully compressed!")

# Usage Examples
if __name__ == "__main__":
    input_csv = "your_file.csv"  # Change this to your file name
    
    # Choose one of the strategies:
    
    # 1. Remove unnecessary columns
    # compress_csv_remove_columns(input_csv, "compressed_v1.csv", 
    #                             columns_to_keep=['col1', 'col2', 'col3'])
    
    # 2. Sample 50% of rows
    # compress_csv_sample(input_csv, "compressed_v2.csv", fraction=0.5)
    
    # 3. Remove duplicates
    # compress_csv_duplicates(input_csv, "compressed_v3.csv")
    
    # 4. Gzip compression (BEST for keeping all data)
    # compress_csv_gzip(input_csv, "compressed_v4.csv.gz")
    
    # 5. Reduce precision
    # compress_csv_precision(input_csv, "compressed_v5.csv", decimal_places=2)
    
    # 6. Optimize data types
    # compress_csv_datatypes(input_csv, "compressed_v6.csv")
    
    # 7. Convert to Parquet
    # compress_csv_parquet(input_csv, "compressed_v7.parquet")
    
    # 8. RECOMMENDED: Combined approach
    compress_csv_combined(input_csv, "compressed_final.csv")
