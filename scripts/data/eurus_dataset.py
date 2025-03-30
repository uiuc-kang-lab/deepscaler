from datasets import load_dataset
import os

# Define data sources
MATH_SOURCES = [
    'numina_olympiads',
    'numina_synthetic_math',
    'numina_cn_k12',
    'numina_synthetic_amc',
    'numina_aops_forum',
    'numina_amc_aime'
]

CODING_SOURCES = [
    'taco',
    'codecontests',
    'codeforces',
    'apps'
]

def filter_by_sources(dataset, sources):
    return dataset.filter(lambda x: x['data_source'] in sources)

def main():
    # Load dataset
    ds = load_dataset("PRIME-RL/Eurus-2-RL-Data")
    
    # Create output directory if it doesn't exist
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each split
    for split in ['train', 'validation']:
        # Filter math and coding data
        math_data = filter_by_sources(ds[split], MATH_SOURCES)
        coding_data = filter_by_sources(ds[split], CODING_SOURCES)
        
        # Save as parquet files
        math_data.to_parquet(f"{output_dir}/eurus_math_{split}.parquet")
        coding_data.to_parquet(f"{output_dir}/eurus_coding_{split}.parquet")
        
        print(f"Processed {split} split:")
        print(f"Math data: {len(math_data)} rows")
        print(f"Coding data: {len(coding_data)} rows")

if __name__ == "__main__":
    main()
