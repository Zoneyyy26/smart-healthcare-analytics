from __future__ import annotations

import sys

REQUIRED = [
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "sklearn",
    "networkx",
    "flask",
    "joblib",
]


def check_packages() -> list[str]:
    missing: list[str] = []
    for package_name in REQUIRED:
        try:
            __import__(package_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print("Install them using: pip install -r requirements.txt")
    else:
        print("[OK] All required packages are installed.")
    return missing


def print_versions() -> None:
    import numpy
    import pandas
    import sklearn

    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Pandas version: {pandas.__version__}")
    print(f"Numpy version: {numpy.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")


if __name__ == "__main__":
    missing_packages = check_packages()
    print_versions()
    if missing_packages:
        raise SystemExit(1)
    print("\nEnvironment check complete.")
