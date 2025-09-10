import config
import requests
import time
import random
try:
    import logger_service
except Exception:  # pragma: no cover
    logger_service = None

DEFAULT_MAX_RETRIES = 3
BASE_BACKOFF = 0.4  # seconds
PARALLEL_ENDPOINT_PRIMARY = "https://api.parallel.ai/v1beta/search"
PARALLEL_ENDPOINT_FALLBACK = "https://api.parallel.ai/v1/search"  # legacy (if still enabled)

def search(query: str, max_retries: int = DEFAULT_MAX_RETRIES, max_results: int = 5, max_chars_per_result: int = 1500) -> list:
    """Search Parallel.ai with simple retry.

    Uses new v1beta Search API (objective + search_queries) and falls back to legacy endpoint if needed.
    Normalizes results to a list of dicts: title, snippet, link.
    """
    if not config.PARALLEL_AI_API_KEY:
        if logger_service:
            logger_service.log_event("search_skip_no_key", query=query)
        return []
    headers = {"x-api-key": f"{config.PARALLEL_AI_API_KEY}"}
    attempt = 0
    endpoint = PARALLEL_ENDPOINT_PRIMARY
    while attempt < max_retries:
        try:
            payload = {
                "objective": query,
                "search_queries": [query],
                "processor": "base",
                "max_results": max_results,
                "max_chars_per_result": max_chars_per_result
            }
            response = requests.post(endpoint, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            raw_results = response.json().get("results", [])
            normalized = []
            for r in raw_results:
                # New API returns url/title/excerpts[]; legacy may return title/snippet/link
                title = r.get("title") or r.get("title", "N/A")
                link = r.get("url") or r.get("link") or r.get("sourceURL") or "N/A"
                if "excerpts" in r and isinstance(r.get("excerpts"), list):
                    snippet = " ".join(r["excerpts"])[:5000]
                else:
                    snippet = r.get("snippet") or r.get("description") or ""
                normalized.append({
                    "title": title or "N/A",
                    "snippet": snippet or "N/A",
                    "link": link
                })
            return normalized
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            attempt += 1
            if attempt >= max_retries:
                if logger_service:
                    logger_service.log_event("search_fail", query=query, attempt=attempt, error=type(e).__name__)
                return []
            sleep_for = BASE_BACKOFF * (2 ** (attempt - 1)) + random.uniform(0, 0.2)
            time.sleep(sleep_for)
        except requests.exceptions.HTTPError as e:
            # Retry only on 5xx
            status = getattr(e.response, 'status_code', None)
            # If 404 on primary endpoint once, attempt legacy fallback then continue
            if status == 404 and endpoint == PARALLEL_ENDPOINT_PRIMARY:
                endpoint = PARALLEL_ENDPOINT_FALLBACK
                if logger_service:
                    logger_service.log_event("search_endpoint_fallback", query=query)
                # do not count this as a retry attempt; continue loop
                continue
            if status and 500 <= status < 600:
                attempt += 1
                if attempt >= max_retries:
                    if logger_service:
                        logger_service.log_event("search_fail_http_5xx", query=query, status=status, attempt=attempt)
                    return []
                sleep_for = BASE_BACKOFF * (2 ** (attempt - 1)) + random.uniform(0, 0.2)
                time.sleep(sleep_for)
                continue
            # Propagate non-retryable errors explicitly
            if logger_service:
                # Include small body snippet for diagnostics
                body_snippet = None
                try:
                    body_snippet = e.response.text[:200] if e.response and e.response.text else None
                except Exception:
                    body_snippet = None
                logger_service.log_event("search_http_error", query=query, status=status, error=str(e), body=body_snippet)
            raise e
    # Exhausted attempts without triggering a terminal return inside loop
    return []
