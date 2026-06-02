import os
import argparse
import pandas as pd
from src.data.generate_mock_data import create_mock_data
from src.models.trainer import run_unified_pipeline

def main():
    parser = argparse.ArgumentParser(description="Arknights Operator Strength NN Training Pipeline")
    parser.add_argument(
        "--profession", 
        type=str, 
        default="all", 
        choices=["all", "Vanguard", "Guard", "Defender", "Sniper", "Caster", "Medic", "Supporter", "Specialist"],
        help="Specify which profession model to train, or 'all' to train models for all professions."
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="data/raw/operators.csv", 
        help="Path to the raw CSV dataset."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=120, 
        help="Max number of epochs to train."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Mini-batch size."
    )
    args = parser.parse_args()
    
    # 1. Ensure raw data exists
    if not os.path.exists(args.data_path):
        print(f"Data file not found at '{args.data_path}'. Generating realistic mock dataset...")
        create_mock_data(output_path=args.data_path)
    else:
        print(f"Using existing data file at '{args.data_path}'.")
        
    # Check for CUDA availability
    device = "cuda" if torch_cuda_is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    
    # 2. Run training pipeline for the unified single model
    try:
        res = run_unified_pipeline(
            data_path=args.data_path, 
            epochs=args.epochs, 
            batch_size=args.batch_size,
            device=device
        )
        
        # 3. Print Summary
        print("\n" + "="*50)
        print("               TRAINING PIPELINE SUMMARY")
        print("="*50)
        print(f"Model: UNIFIED SINGLE MULTI-PROFESSION MODEL")
        print(f"Test Accuracy: {res['test_acc']:.4%}")
        print(f"Status: SUCCESS")
        print("="*50)
        print(f"Successfully trained unified model saved in './models/unified_model.pth'.")
        print("==================================================")
    except Exception as e:
        print(f"Failed to train unified model. Error: {str(e)}")
        import traceback
        traceback.print_exc()

def torch_cuda_is_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

if __name__ == "__main__":
    main()
