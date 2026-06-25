import os
import sys

# Make the repo-root modules (strategy.py, risk.py, …) importable when pytest
# is invoked from anywhere. strategy/risk are pure and import no credentials.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
