"""HTTP client for the Kayba hosted API."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests


class KaybaAPIError(Exception):
    """Structured error from the Kayba API."""

    def __init__(self, code: str, message: str, status_code: int = 0):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(f"[{code}] {message}")


DEFAULT_BASE_URL = "https://use.kayba.ai/api"


class KaybaClient:
    """HTTP client for the Kayba hosted API.

    Args:
        api_key: Kayba API key. Falls back to KAYBA_API_KEY env var.
        base_url: API base URL. Falls back to KAYBA_API_URL env var,
                  then to https://use.kayba.ai/api.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("KAYBA_API_KEY", "")
        if not self.api_key:
            raise KaybaAPIError(
                "AUTH_MISSING",
                "No API key provided. Set KAYBA_API_KEY or pass --api-key.",
            )
        self.base_url = (
            base_url or os.environ.get("KAYBA_API_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {self.api_key}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Send a request and return parsed JSON, raising on API errors."""
        url = f"{self.base_url}{path}"
        resp = self.session.request(method, url, json=json, params=params)

        if resp.status_code >= 400:
            try:
                body = resp.json()
                err = body.get("error", {})
                raise KaybaAPIError(
                    code=err.get("code", "UNKNOWN"),
                    message=err.get("message", resp.text),
                    status_code=resp.status_code,
                )
            except (ValueError, KeyError):
                raise KaybaAPIError(
                    code="HTTP_ERROR",
                    message=resp.text,
                    status_code=resp.status_code,
                )

        if resp.status_code == 204:
            return {}
        return resp.json()

    # -- Traces --

    def upload_traces(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Upload trace files.

        Args:
            traces: List of dicts with keys: filename, content, fileType.
        """
        return self._request("POST", "/traces", json={"traces": traces})

    def list_traces(self) -> Dict[str, Any]:
        """List all traces (metadata only, no content)."""
        return self._request("GET", "/traces")

    def get_trace(self, trace_id: str) -> Dict[str, Any]:
        """Get a single trace with full content."""
        return self._request("GET", f"/traces/{trace_id}")

    def get_traces(self, trace_ids: List[str]) -> Dict[str, Any]:
        """Batch get traces by IDs (with content)."""
        return self._request("POST", "/traces/batch", json={"ids": trace_ids})

    def delete_trace(self, trace_id: str) -> Dict[str, Any]:
        """Delete a single trace."""
        return self._request("DELETE", f"/traces/{trace_id}")

    def delete_traces(self, trace_ids: List[str]) -> Dict[str, Any]:
        """Delete multiple traces."""
        results = []
        errors = []
        for tid in trace_ids:
            try:
                self.delete_trace(tid)
                results.append(tid)
            except KaybaAPIError as e:
                errors.append({"id": tid, "error": str(e)})
        return {"deleted": results, "errors": errors}

    # -- Insights --

    def generate_insights(
        self,
        *,
        trace_ids: Optional[List[str]] = None,
        model: Optional[str] = None,
        epochs: Optional[int] = None,
        reflector_mode: Optional[str] = None,
        anthropic_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start async insight generation."""
        body: Dict[str, Any] = {}
        if trace_ids:
            body["traceIds"] = trace_ids
        if model:
            body["model"] = model
        if epochs is not None:
            body["epochs"] = epochs
        if reflector_mode:
            body["reflectorMode"] = reflector_mode
        if anthropic_key:
            body["anthropicApiKey"] = anthropic_key
        return self._request("POST", "/insights/generate", json=body)

    def list_insights(
        self,
        *,
        status: Optional[str] = None,
        section: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List insights, optionally filtered."""
        params: Dict[str, str] = {}
        if status:
            params["status"] = status
        if section:
            params["section"] = section
        return self._request("GET", "/insights", params=params or None)

    def triage_insight(
        self,
        insight_id: str,
        status: str,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Accept or reject a single insight."""
        body: Dict[str, Any] = {"status": status}
        if note:
            body["note"] = note
        return self._request("PATCH", f"/insights/{insight_id}", json=body)

    # -- Jobs --

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Get job status."""
        return self._request("GET", f"/jobs/{job_id}")

    def materialize_job(self, job_id: str) -> Dict[str, Any]:
        """Materialize completed job results into the skillbook."""
        return self._request("POST", f"/jobs/{job_id}")

    # -- Prompts --

    def generate_prompt(
        self,
        *,
        insight_ids: Optional[List[str]] = None,
        label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a prompt from accepted insights."""
        body: Dict[str, Any] = {}
        if insight_ids:
            body["insightIds"] = insight_ids
        if label:
            body["label"] = label
        return self._request("POST", "/prompts/generate", json=body)

    def list_prompts(self) -> Dict[str, Any]:
        """List all prompt versions."""
        return self._request("GET", "/prompts")

    def get_prompt(self, prompt_id: str) -> Dict[str, Any]:
        """Get a specific prompt by ID."""
        return self._request("GET", f"/prompts/{prompt_id}")
