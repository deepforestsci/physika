import sys
from pathlib import Path

# Add physika/ to sys.path so tests can import codegen, utils, etc.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
