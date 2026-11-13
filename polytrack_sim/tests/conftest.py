import os
import sys

# Make `import sim`, `import render2d` and `import examples...` work from any cwd.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))
