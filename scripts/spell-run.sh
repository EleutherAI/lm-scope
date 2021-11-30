spell run \
    --machine-type gpu \
    --pip-req requirements.txt \
    --force \
    --mount azblob/rgsbstorage_lm-scope-data:/mnt/data \
    --env AZ_ACCOUNT_NAME="$AZ_ACCOUNT_NAME" \
    --env AZ_ACCOUNT_KEY="$AZ_ACCOUNT_KEY" \
    --env AZ_CONTAINER_NAME="$AZ_CONTAINER_NAME" \
    "python scripts/main.py --num-workers 3 --dataset-offset $1 --dataset-limit $2"

