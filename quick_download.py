#!/usr/bin/env python3
"""
Quick Dataset Downloader

A simple script to quickly download parquet datasets.
Usage: python quick_download.py
"""

import requests
from pathlib import Path

def download_parquet_dataset(dataset_name: str, output_dir: str = "./data"):
    """
    Quick function to download a parquet dataset
    
    Args:
        dataset_name (str): Name of the dataset to download
        output_dir (str): Directory to save the file
    """
    
    # Dataset URLs mapping
    datasets = {
        'cotton80': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/ufgvc/cotton80_dataset.parquet?download=true',
            'filename': 'cotton80_dataset.parquet'
        },
        'soybean': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/ufgvc/soybean_dataset.parquet?download=true',
            'filename': 'soybean_dataset.parquet'
        },
        'soy_ageing_r1': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/ufgvc/soy_ageing_R1_dataset.parquet?download=true',
            'filename': 'soy_ageing_R1_dataset.parquet'
        },
        'soy_ageing_r3': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/ufgvc/soy_ageing_R3_dataset.parquet?download=true',
            'filename': 'soy_ageing_R3_dataset.parquet'
        },
        'soy_ageing_r4': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/ufgvc/soy_ageing_R4_dataset.parquet?download=true',
            'filename': 'soy_ageing_R4_dataset.parquet'
        },
        'soy_ageing_r5': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/ufgvc/soy_ageing_R5_dataset.parquet?download=true',
            'filename': 'soy_ageing_R5_dataset.parquet'
        },
        'soy_ageing_r6': {
            'url': 'https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/ufgvc/soy_ageing_R6_dataset.parquet?download=true',
            'filename': 'soy_ageing_R6_dataset.parquet'
        }
    }
    
    if dataset_name not in datasets:
        print(f"❌ Dataset '{dataset_name}' not found!")
        print(f"Available datasets: {list(datasets.keys())}")
        return None
    
    # Setup paths
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dataset_config = datasets[dataset_name]
    filepath = output_path / dataset_config['filename']
    
    # Check if already exists
    if filepath.exists():
        print(f"✅ Dataset '{dataset_name}' already exists at: {filepath}")
        return str(filepath)
    
    # Download
    print(f"📥 Downloading {dataset_name}...")
    try:
        response = requests.get(dataset_config['url'], stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\r📊 Progress: {progress:.1f}% ({downloaded_size//1024//1024}MB)", end="")
        
        print(f"\n✅ Download completed: {filepath}")
        return str(filepath)
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        if filepath.exists():
            filepath.unlink()
        return None

def main():
    """Interactive main function"""
    datasets = ['cotton80', 'soybean', 'soy_ageing_r1', 'soy_ageing_r3', 
                'soy_ageing_r4', 'soy_ageing_r5', 'soy_ageing_r6']
    
    print("🌱 UFGVC Dataset Downloader")
    print("=" * 30)
    print("Available datasets:")
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i}. {dataset}")
    print()
    
    try:
        choice = input("Enter dataset name or number (or 'all' for all datasets): ").strip()
        
        if choice.lower() == 'all':
            print("📦 Downloading all datasets...")
            for dataset in datasets:
                download_parquet_dataset(dataset)
                print()
        elif choice.isdigit() and 1 <= int(choice) <= len(datasets):
            dataset_name = datasets[int(choice) - 1]
            download_parquet_dataset(dataset_name)
        elif choice in datasets:
            download_parquet_dataset(choice)
        else:
            print("❌ Invalid choice!")
            
    except KeyboardInterrupt:
        print("\n❌ Download cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
