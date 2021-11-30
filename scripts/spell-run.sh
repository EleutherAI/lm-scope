spell run \
    --machine-type gpu-on-demand \
    --pip-req requirements.txt \
    --force \
    --mount azblob/rgsbstorage_lm-scope-data:/mnt/data \
    "python scripts/main.py --dataset-limit 1"

