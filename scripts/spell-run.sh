spell run \
    --machine-type gpu \
    --pip-req requirements.txt \
    --force \
    --mount azblob/rgsbstorage_lm-scope-data:/mnt/data \
    "python scripts/main.py"

