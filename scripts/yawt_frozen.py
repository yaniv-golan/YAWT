#!/usr/bin/env python3
import os
import sys

# Add the src directory to Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, src_path)

# Now import and run the main function
from yawt.main import main

if __name__ == '__main__':
    main() 