"""CheckoutManager: atomic problem leasing via HuggingFace Hub.

Agents call checkout() to claim N problems from bsmith925/cnake-charmer-stack-v2.
The registry (data/checkouts.jsonl on HF) is updated with optimistic locking:
if two agents try to commit simultaneously, the loser retries automatically.
"""

import random
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from cnake_charmer.checkout.hf import read_registry, write_registry

REGISTRY_REPO = "bsmith925/cnake-charmer-stack-v2"
_CANDIDATES_CACHE = Path.home() / ".cache" / "cnake_charmer" / "candidates.parquet"
_CACHE_TTL_SECONDS = 3600  # 1 hour


def _now() -> datetime:
    return datetime.now(UTC)


def _load_candidates(repo_id: str, token: str | None) -> list[dict]:
    """Load the candidates config from HF, with a 1-hour local cache."""
    if (
        _CANDIDATES_CACHE.exists()
        and (time.time() - _CANDIDATES_CACHE.stat().st_mtime) < _CACHE_TTL_SECONDS
    ):
        import pandas as pd

        return pd.read_parquet(_CANDIDATES_CACHE).to_dict("records")

    local = hf_hub_download(
        repo_id=repo_id,
        filename="data/candidates/train-00000-of-00001.parquet",
        repo_type="dataset",
        token=token,
    )
    # Try standard HF parquet naming; fall back to datasets API if not found
    try:
        import pandas as pd

        df = pd.read_parquet(local)
    except Exception:
        from datasets import load_dataset

        ds = load_dataset(repo_id, "candidates", split="train", token=token)
        import pandas as pd

        df = ds.to_pandas()

    _CANDIDATES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_CANDIDATES_CACHE)
    return df.to_dict("records")


def _load_candidates_reliable(repo_id: str, token: str | None) -> list[dict]:
    """Load candidates using datasets library (handles parquet sharding)."""
    if (
        _CANDIDATES_CACHE.exists()
        and (time.time() - _CANDIDATES_CACHE.stat().st_mtime) < _CACHE_TTL_SECONDS
    ):
        import pandas as pd

        return pd.read_parquet(_CANDIDATES_CACHE).to_dict("records")

    import pandas as pd
    from datasets import load_dataset

    ds = load_dataset(repo_id, "candidates", split="train", token=token)
    df = ds.to_pandas()
    _CANDIDATES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_CANDIDATES_CACHE)
    return df.to_dict("records")


class CheckoutManager:
    """Manage problem checkouts via the HF Hub registry."""

    def __init__(
        self,
        registry_repo: str = REGISTRY_REPO,
        token: str | None = None,
        expiry_hours: int = 48,
        max_retries: int = 5,
    ):
        self.registry_repo = registry_repo
        self.token = token
        self.expiry_hours = expiry_hours
        self.max_retries = max_retries

    def _get_token(self) -> str | None:
        if self.token:
            return self.token
        try:
            from huggingface_hub import HfFolder

            return HfFolder.get_token()
        except Exception:
            return None

    def _expire_stale(self, records: list[dict]) -> list[dict]:
        now = _now()
        for r in records:
            if r["status"] == "checked_out":
                expires = datetime.fromisoformat(r["expires_at"])
                if expires.tzinfo is None:
                    expires = expires.replace(tzinfo=UTC)
                if expires < now:
                    r["status"] = "abandoned"
        return records

    def checkout(self, worker_id: str, n: int) -> list[dict]:
        """Atomically claim N unchecked-out candidates.

        Returns list of candidate rows (including content) for the claimed problems.
        Raises ValueError if no candidates are available.
        """
        token = self._get_token()
        candidates = _load_candidates_reliable(self.registry_repo, token)
        candidates_by_id = {r["blob_id"]: r for r in candidates}

        for attempt in range(self.max_retries):
            sha, records = read_registry(self.registry_repo, token)
            records = self._expire_stale(records)

            claimed = {r["blob_id"] for r in records if r["status"] in ("checked_out", "completed")}
            available = [r for r in candidates if r["blob_id"] not in claimed]

            if not available:
                raise ValueError("No candidates available for checkout.")

            selected = random.sample(available, min(n, len(available)))
            now = _now()
            expires = now + timedelta(hours=self.expiry_hours)

            new_records = [
                {
                    "blob_id": row["blob_id"],
                    "worker_id": worker_id,
                    "claimed_at": now.isoformat(),
                    "expires_at": expires.isoformat(),
                    "status": "checked_out",
                    "problem_id": None,
                    "completed_at": None,
                    "notes": None,
                }
                for row in selected
            ]
            updated = records + new_records

            try:
                write_registry(
                    repo_id=self.registry_repo,
                    token=token,
                    records=updated,
                    parent_commit=sha,
                    commit_message=f"checkout: {worker_id} claims {len(new_records)} problems",
                )
                return [candidates_by_id[r["blob_id"]] for r in new_records]
            except HfHubHTTPError as e:
                if e.response is not None and e.response.status_code == 412:
                    sleep = 0.5 * (2**attempt) + random.uniform(0, 0.5)
                    time.sleep(sleep)
                    continue
                raise

        raise RuntimeError(f"Failed to checkout after {self.max_retries} retries due to conflicts.")

    def _update_record(
        self,
        blob_id: str,
        worker_id: str,
        updates: dict,
        commit_message: str,
    ) -> None:
        """Find a record by blob_id and apply updates. Retries on conflict."""
        token = self._get_token()

        for attempt in range(self.max_retries):
            sha, records = read_registry(self.registry_repo, token)
            records = self._expire_stale(records)

            found = False
            for r in records:
                if r["blob_id"] == blob_id and r["worker_id"] == worker_id:
                    r.update(updates)
                    found = True
                    break

            if not found:
                raise ValueError(
                    f"No active checkout found for blob_id={blob_id!r} worker={worker_id!r}"
                )

            try:
                write_registry(
                    repo_id=self.registry_repo,
                    token=token,
                    records=records,
                    parent_commit=sha,
                    commit_message=commit_message,
                )
                return
            except HfHubHTTPError as e:
                if e.response is not None and e.response.status_code == 412:
                    sleep = 0.5 * (2**attempt) + random.uniform(0, 0.5)
                    time.sleep(sleep)
                    continue
                raise

        raise RuntimeError(f"Failed to update checkout after {self.max_retries} retries.")

    def complete(
        self,
        blob_id: str,
        worker_id: str,
        problem_id: str,
        verify_exists: bool = True,
    ) -> None:
        """Mark a checkout as completed.

        verify_exists checks that problem_id is present in the local repo via
        discover_pairs(). Pass verify_exists=False on machines without repo access.
        """
        if verify_exists:
            from cnake_data.loader import discover_pairs

            known = {p.problem_id for p in discover_pairs()}
            if problem_id not in known:
                raise ValueError(
                    f"problem_id {problem_id!r} not found in discover_pairs(). "
                    "Use --no-verify if working outside the repo."
                )

        self._update_record(
            blob_id=blob_id,
            worker_id=worker_id,
            updates={
                "status": "completed",
                "problem_id": problem_id,
                "completed_at": _now().isoformat(),
            },
            commit_message=f"complete: {worker_id} finished {problem_id}",
        )

    def fail(self, blob_id: str, worker_id: str, notes: str = "") -> None:
        """Mark a checkout as failed (problem stays claimable)."""
        self._update_record(
            blob_id=blob_id,
            worker_id=worker_id,
            updates={"status": "failed", "notes": notes or None},
            commit_message=f"fail: {worker_id} could not complete {blob_id[:8]}",
        )

    def abandon(self, blob_id: str, worker_id: str) -> None:
        """Return a problem to the pool immediately."""
        self._update_record(
            blob_id=blob_id,
            worker_id=worker_id,
            updates={"status": "abandoned"},
            commit_message=f"abandon: {worker_id} returns {blob_id[:8]}",
        )

    def status(self, worker_id: str | None = None) -> dict:
        """Return aggregate checkout statistics."""
        token = self._get_token()
        _, records = read_registry(self.registry_repo, token)
        records = self._expire_stale(records)

        candidates = _load_candidates_reliable(self.registry_repo, token)
        total = len(candidates)

        completed_ids = {r["blob_id"] for r in records if r["status"] == "completed"}
        checked_out_ids = {r["blob_id"] for r in records if r["status"] == "checked_out"}
        failed_ids = {r["blob_id"] for r in records if r["status"] == "failed"}

        unavailable = completed_ids | checked_out_ids
        available = total - len(unavailable)

        by_worker: dict[str, dict] = {}
        for r in records:
            w = r["worker_id"]
            if w not in by_worker:
                by_worker[w] = {"checked_out": 0, "completed": 0, "failed": 0}
            if r["status"] in by_worker[w]:
                by_worker[w][r["status"]] += 1

        result = {
            "total_candidates": total,
            "available": available,
            "checked_out": len(checked_out_ids),
            "completed": len(completed_ids),
            "failed": len(failed_ids),
            "by_worker": by_worker,
        }

        if worker_id:
            result["by_worker"] = {worker_id: by_worker.get(worker_id, {})}

        return result

    def list_checked_out(self, worker_id: str | None = None) -> list[dict]:
        """Return active checked_out records, optionally filtered by worker."""
        token = self._get_token()
        _, records = read_registry(self.registry_repo, token)
        records = self._expire_stale(records)

        active = [r for r in records if r["status"] == "checked_out"]
        if worker_id:
            active = [r for r in active if r["worker_id"] == worker_id]
        return active
