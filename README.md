# SharingSchemes

Implementations for secret sharing schemes

## Implementations

- Asmuth-Bloom
- Blakley
- Feldman
- Pedersen
- Shamir

## Requirements

- Python 3
- venv (optional)

## Setup

``` sh
# Create environment
cd ~/venvs # Store in venvs
python -m venv csci-ga-3033-107-ss
source ~/venvs/csci-ga-3033-107-ss/bin/activate
deactivate # Deactivate from within env
```

Install dependencies:

``` sh
pip install -r requirements.txt
```

## Running

``` sh
# Example running Shamir
python shamirsecretsharing.py
```

To regenerate the data and graphs from scratch:
* Run generate_csv.py (Note that this step will take very long)
* Run graph_results.py


## Working

Save dependencies:

``` sh
pip freeze > requirements.txt
```
