"""Single source of truth for project root. Override with PROJECT_ROOT env var."""
import os
PROJECT_ROOT = os.environ.get("PROJECT_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(PROJECT_ROOT, "data")
