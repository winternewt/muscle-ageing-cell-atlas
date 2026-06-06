#!/usr/bin/env python3
"""
Simple runner for dataset validation with dependency handling
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check and install required dependencies"""
    required_packages = ['datasets', 'pandas', 'pyarrow']
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} not found, installing...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def main():
    """Main runner function"""
    print("🔧 Checking dependencies...")
    check_dependencies()
    
    print("\n🚀 Running dataset validation...")
    try:
        # Import and run the validation
        from validate_dataset_loading import main as validate_main
        validate_main()
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 