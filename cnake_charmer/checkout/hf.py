"""HuggingFace Hub I/O for the checkout registry.

The registry is a JSONL file stored at data/checkouts.jsonl inside the
bsmith925/cnake-charmer-stack-v2 dataset repo. Reads return the current
commit SHA alongside the records so callers can use optimistic locking:
write_registry passes parent_commit back to create_commit, which fails with
HTTP 412 if another agent committed in the meantime.
"""

import io
import json

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

REGISTRY_PATH = "data/checkouts.jsonl"
_api = HfApi()


def read_registry(repo_id: str, token: str | None = None) -> tuple[str, list[dict]]:
    """Download checkouts.jsonl and return (current_commit_sha, records).

    Returns an empty records list if the file doesn't exist yet.
    """
    info = _api.repo_info(repo_id, repo_type="dataset", token=token)
    sha = info.sha

    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=REGISTRY_PATH,
            repo_type="dataset",
            token=token,
            force_download=True,
        )
        records = []
        with open(local) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return sha, records
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return sha, []
        raise


def write_registry(
    repo_id: str,
    token: str | None,
    records: list[dict],
    parent_commit: str,
    commit_message: str,
) -> None:
    """Upload checkouts.jsonl with optimistic locking.

    Raises HfHubHTTPError (HTTP 412) if the repo has moved since parent_commit,
    signalling the caller to re-read and retry.
    """
    lines = [json.dumps(r, default=str) for r in records]
    content = ("\n".join(lines) + "\n").encode()

    _api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(path_in_repo=REGISTRY_PATH, path_or_fileobj=io.BytesIO(content))
        ],
        commit_message=commit_message,
        token=token,
        parent_commit=parent_commit,
    )


def init_registry(repo_id: str, token: str | None = None) -> None:
    """Create an empty checkouts.jsonl on the HF repo if it doesn't exist."""
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=REGISTRY_PATH,
            repo_type="dataset",
            token=token,
        )
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            _api.upload_file(
                path_or_fileobj=io.BytesIO(b""),
                path_in_repo=REGISTRY_PATH,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                commit_message="chore: initialize empty checkout registry",
            )
        else:
            raise
