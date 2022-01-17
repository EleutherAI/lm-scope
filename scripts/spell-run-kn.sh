spell run \
    --machine-type V100x8-1tb-disk \
    --pip-req requirements.txt \
    --mount azblob/rgsbstorage_lm-scope-data:/mnt/data \
    "python scripts/kn.py"
