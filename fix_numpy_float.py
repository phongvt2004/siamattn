#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to fix deprecated np.float usage for NumPy 2.0 compatibility
"""

import os
import re
import glob

def fix_numpy_float(file_path):
    """Fix np.float to np.float32 in a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace np.float (not followed by digit) with np.float32
        # Pattern: np.float followed by ) or ] or space or newline
        content = re.sub(r'np\.float([^0-9])', r'np.float32\1', content)
        
        # Replace dtype=np.float with dtype=np.float32
        content = re.sub(r'dtype=np\.float([^0-9])', r'dtype=np.float32\1', content)
        
        # Replace dtype=np.float) with dtype=np.float32)
        content = re.sub(r'dtype=np\.float\)', r'dtype=np.float32)', content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Fixed: {file_path}")
            return True
        return False
    except Exception as e:
        print(f"✗ Error fixing {file_path}: {e}")
        return False

def main():
    """Fix all Python files"""
    print("Fixing deprecated np.float usage...")
    print("=" * 50)
    
    # Find all Python files
    python_files = []
    
    # Files in pysot directory
    python_files.extend(glob.glob('pysot/**/*.py', recursive=True))
    
    # Files in training_dataset (optional, mainly for reference)
    python_files.extend(glob.glob('training_dataset/**/*.py', recursive=True))
    
    fixed_count = 0
    for file_path in python_files:
        if fix_numpy_float(file_path):
            fixed_count += 1
    
    print("=" * 50)
    print(f"Fixed {fixed_count} files")
    print("Done!")

if __name__ == '__main__':
    main()

