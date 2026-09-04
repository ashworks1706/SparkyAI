"""S3-compatible client for raw snapshots and artifacts."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from scraper.core.settings import settings


@lru_cache(maxsize=1)
def client() -> Any:
    s = settings().object_store
    return boto3.client(
        "s3",
        endpoint_url=s.endpoint,
        aws_access_key_id=s.access_key,
        aws_secret_access_key=s.secret_key.get_secret_value(),
        region_name=s.region,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def ensure_bucket() -> None:
    """Creates the bucket when it is missing. Any other error (credentials, network) propagates."""
    bucket = settings().object_store.bucket
    try:
        client().head_bucket(Bucket=bucket)
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchBucket"):
            raise
        client().create_bucket(Bucket=bucket)


def put_snapshot(key: str, body: bytes, content_type: str) -> str:
    """Stores a raw page. Returns the object key."""
    ensure_bucket()
    client().put_object(
        Bucket=settings().object_store.bucket, Key=key, Body=body, ContentType=content_type
    )
    return key
