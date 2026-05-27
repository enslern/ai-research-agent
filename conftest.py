"""
conftest.py — adds the project root to sys.path so pytest can resolve
package imports without requiring an editable install.
"""

import sys
import os

# Ensure the project root is always on the path
sys.path.insert(0, os.path.dirname(__file__))
