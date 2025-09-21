#!/usr/bin/env python3
"""
NASA InSight Downloader using PDS Search API (v1) + ds-view fallback.

Fixes:
- Use start/limit (not page/page_size) to avoid 400 responses on MCP backend.
- Fail fast on 400/422/500 and fall back to ds-view parsing immediately.
"""

import os
import re
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urljoin, quote

import requests

PDS_API_ROOT = "https://pds.nasa.gov/api/search/1"
PDS_SEARCH_PRODUCTS = f"{PDS_API_ROOT}/products"
PDS_SEARCH_RECORDS = f"{PDS_API_ROOT}/records"
DS_VIEW_BASE = "https://pds.nasa.gov/ds-view/pds/viewProduct.jsp?identifier="

class NASAInSightDownloader:
    def __init__(self, download_dir="nasa_insight_data"):
        self.download_dir = Path(download_dir)
        self.twins_dir = self.download_dir / "twins_data"
        self.ps_dir = self.download_dir / "ps_data"
        self.download_dir.mkdir(exist_ok=True, parents=True)
        self.twins_dir.mkdir(exist_ok=True, parents=True)
        self.ps_dir.mkdir(exist_ok=True, parents=True)

        self.urn_files = {
            "twins": "twins_2018_2020_urns.txt",
            "ps": "ps_2018_2020_urns.txt",
        }

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "NASA-InSight-Downloader/2.2 (+https://pds.nasa.gov/)",
            "Accept": "application/json",
        })

        self.max_retries = 3
        self.retry_delay = 2.0
        self.request_delay = 0.4  # politeness

        self.stats = {
            "products_total": 0,
            "products_ok": 0,
            "products_failed": 0,
            "files_downloaded": 0,
            "files_skipped": 0,
            "files_failed": 0,
            "total_bytes": 0,
        }

    # ---------- Logging ----------
    def log(self, msg, level="INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {level}: {msg}")
        sys.stdout.flush()

    # ---------- Utilities ----------
    def safe_dirname(self, urn: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", urn)[:180]

    def load_urns(self, filepath: Path):
        urns = []
        if not filepath.exists():
            self.log(f"URN file not found: {filepath}", "ERROR")
            return urns
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    urns.append(s)
        self.log(f"Loaded {len(urns)} URNs from {filepath}", "SUCCESS")
        return urns

    def _get_json_once(self, url, params=None, timeout=30):
        # Single GET; returns (data, status_code or None)
        try:
            r = self.session.get(url, params=params, timeout=timeout, allow_redirects=True)
            sc = r.status_code
            if sc == 404:
                return None, sc
            if sc in (400, 401, 403, 422, 500, 502, 503, 504):
                # treat as fatal for this endpoint/param combo
                return None, sc
            r.raise_for_status()
            return r.json(), sc
        except requests.RequestException as e:
            self.log(f"HTTP error for {url}: {e}", "WARNING")
            return None, None

    def _extract_urls_from_properties(self, props: dict):
        urls = []

        def add_from_value(val):
            if not val:
                return
            if isinstance(val, str):
                if val.startswith("http"):
                    urls.append(val)
            elif isinstance(val, dict):
                href = val.get("href") or val.get("url") or val.get("URL")
                if href and isinstance(href, str) and href.startswith("http"):
                    urls.append(href)
            elif isinstance(val, list):
                for v in val:
                    add_from_value(v)

        for key in [
            "ops:Label_File",
            "ops:Data_File",
            "ops:Data_Files",
            "ops:Download_URL",
            "ops:URL",
            "ops:files",
        ]:
            if key in props:
                add_from_value(props[key])

        # Deduplicate preserving order
        seen = set()
        out = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out

    def _search_api_for_urn(self, urn: str):
        # API v1 uses start/limit. Try products, then records. Use only 'q' to avoid backend quirks.
        params = {
            "q": f'identifier:"{urn}"',
            "fields": "identifier,title,ops:Label_File,ops:Data_File,ops:Data_Files,ops:files,ops:Download_URL,ops:URL,links",
            "start": 0,
            "limit": 1,
        }

        # Try /products
        data, sc = self._get_json_once(PDS_SEARCH_PRODUCTS, params=params)
        if sc in (400, 422, 500):
            self.log("PDS API /products returned error; will fall back or try /records.", "DEBUG")
            data = None

        # Try /records if /products didn't return items
        if not data or not isinstance(data, dict) or not data.get("items"):
            data2, sc2 = self._get_json_once(PDS_SEARCH_RECORDS, params=params)
            if sc2 in (400, 422, 500):
                self.log("PDS API /records returned error; will fall back to ds-view.", "DEBUG")
                data2 = None
            data = data2 if data2 else data

        if not (isinstance(data, dict) and data.get("items")):
            return []

        item = data["items"][0]
        props = item.get("properties", {})
        urls = self._extract_urls_from_properties(props)

        # Fallback: inspect item links if no ops:* URLs present
        if not urls and isinstance(item.get("links"), list):
            for link in item["links"]:
                href = link.get("href")
                if isinstance(href, str) and href.startswith("http"):
                    urls.append(href)

        # Clean/dedup
        seen = set()
        cleaned = []
        for u in urls:
            if isinstance(u, str) and u.startswith("http") and u not in seen:
                seen.add(u)
                cleaned.append(u)
        return cleaned

    def parse_ds_view_for_files(self, urn: str):
        # Parse ds-view product page to collect viewFile links for label + data
        product_page = DS_VIEW_BASE + quote(urn, safe="")
        try:
            r = self.session.get(product_page, timeout=30)
            if r.status_code != 200:
                return []
            html = r.text
        except requests.RequestException:
            return []

        # Grab all .../viewFile.jsp?... links
        urls = []
        for match in re.findall(r'href="([^"]+viewFile\.jsp\?[^"]+)"', html, re.IGNORECASE):
            full = urljoin("https://pds.nasa.gov/", match)
            urls.append(full)

        # Deduplicate
        seen = set()
        out = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out

    def resolve_urn_to_file_urls(self, urn: str):
        # Try API first
        urls = self._search_api_for_urn(urn)

        # If API failed or returned nothing, use ds-view fallback
        if not urls:
            self.log("API did not return file URLs; falling back to ds-view parsing.", "WARNING")
            urls = self.parse_ds_view_for_files(urn)

        # Filter out viewer pages if any slipped in (keep viewFile.jsp which redirects to actual files)
        cleaned = []
        for u in urls:
            if "viewProduct.jsp" in u:
                continue
            cleaned.append(u)

        # Dedup
        seen = set()
        final = []
        for u in cleaned:
            if u not in seen:
                seen.add(u)
                final.append(u)
        return final

    # ---------- Download ----------
    def download_url(self, url: str, dest_dir: Path):
        tmp_path = None
        try:
            with self.session.get(url, stream=True, timeout=90, allow_redirects=True) as r:
                r.raise_for_status()

                final_url = r.url
                filename = None

                cd = r.headers.get("Content-Disposition") or r.headers.get("content-disposition")
                if cd:
                    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd)
                    if m:
                        filename = m.group(1).strip().strip('"')

                if not filename:
                    filename = final_url.split("?")[0].rstrip("/").split("/")[-1] or "download.bin"

                dest_path = dest_dir / filename
                if dest_path.exists():
                    self.stats["files_skipped"] += 1
                    self.log(f"SKIP (exists): {dest_path}", "SKIP")
                    return True

                tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

                total = int(r.headers.get("Content-Length", "0") or "0")
                downloaded = 0
                t0 = time.time()
                last_log = t0

                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 64):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if (now - last_log) >= 2.0:
                            if total > 0:
                                pct = (downloaded / total) * 100
                                self.log(f"  {filename}: {downloaded:,}/{total:,} bytes ({pct:.1f}%)", "DEBUG")
                            else:
                                self.log(f"  {filename}: {downloaded:,} bytes", "DEBUG")
                            last_log = now

                os.replace(tmp_path, dest_path)
                dt = time.time() - t0
                rate_kb = (downloaded / 1024.0) / max(dt, 1e-6)
                self.log(f"OK: {filename} ({downloaded:,} bytes) in {dt:.1f}s [{rate_kb:.1f} KB/s]", "SUCCESS")

                self.stats["files_downloaded"] += 1
                self.stats["total_bytes"] += downloaded
                return True

        except requests.RequestException as e:
            self.log(f"Download failed ({url}): {e}", "ERROR")
            self.stats["files_failed"] += 1
            # cleanup partial
            try:
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            return False
        except Exception as e:
            self.log(f"Unexpected error for {url}: {e}", "ERROR")
            self.stats["files_failed"] += 1
            try:
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            return False

    def download_product(self, urn: str, base_dir: Path):
        prod_dir = base_dir / self.safe_dirname(urn)
        prod_dir.mkdir(exist_ok=True, parents=True)

        urls = self.resolve_urn_to_file_urls(urn)
        if not urls:
            self.log(f"No file URLs found for {urn}", "ERROR")
            self.stats["products_failed"] += 1
            return

        ok = False
        for u in urls:
            if "viewProduct.jsp" in u:
                continue
            for attempt in range(1, self.max_retries + 1):
                self.log(f"Downloading ({attempt}/{self.max_retries}): {u}", "DOWNLOAD")
                if self.download_url(u, prod_dir):
                    ok = True
                    break
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)
            time.sleep(self.request_delay)

        if ok:
            self.stats["products_ok"] += 1
        else:
            self.stats["products_failed"] += 1

    def download_from_list(self, data_type: str):
        base_dir = self.twins_dir if data_type == "twins" else self.ps_dir
        urn_file = Path(self.urn_files[data_type])
        urns = self.load_urns(urn_file)
        if not urns:
            self.log(f"No URNs to process for {data_type}", "WARNING")
            return

        self.stats["products_total"] += len(urns)
        t0 = time.time()

        for i, urn in enumerate(urns, start=1):
            elapsed = time.time() - t0
            eta = "calculating..." if i == 1 else str(
                timedelta(seconds=int(elapsed / max(i - 1, 1) * (len(urns) - (i - 1))))
            )
            self.log(f"--- {data_type.upper()} {i}/{len(urns)} | Elapsed {timedelta(seconds=int(elapsed))} | ETA {eta} ---")
            self.log(f"URN: {urn}")
            self.download_product(urn, base_dir)

    def print_summary(self):
        self.log("--- SUMMARY ---")
        self.log(f"Products total:    {self.stats['products_total']}")
        self.log(f"Products ok:       {self.stats['products_ok']}")
        self.log(f"Products failed:   {self.stats['products_failed']}")
        self.log(f"Files downloaded:  {self.stats['files_downloaded']}")
        self.log(f"Files skipped:     {self.stats['files_skipped']}")
        self.log(f"Files failed:      {self.stats['files_failed']}")
        self.log(f"Total data:        {self.stats['total_bytes'] / (1024**2):.2f} MB")
        self.log(f"TWINS dir: {self.twins_dir} ({len(list(self.twins_dir.rglob('*')))} items)")
        self.log(f"PS dir:    {self.ps_dir} ({len(list(self.ps_dir.rglob('*')))} items)")

    def run(self):
        missing = [f for f in self.urn_files.values() if not Path(f).exists()]
        if missing:
            self.log("Missing URN files:", "ERROR")
            for m in missing:
                self.log(f"  - {m}", "ERROR")
            return

        self.download_from_list("twins")
        self.download_from_list("ps")
        self.print_summary()

def main():
    print("NASA InSight Data Downloader (PDS API v1 + ds-view fallback)")
    required = ["twins_2018_2020_urns.txt", "ps_2018_2020_urns.txt"]
    missing = [f for f in required if not Path(f).exists()]
    if missing:
        print("Missing URN files:")
        for m in missing:
            print(f" - {m}")
        print("Please create them first.")
        return

    for f in required:
        with open(f, "r", encoding="utf-8") as fp:
            cnt = sum(1 for line in fp if line.strip() and not line.strip().startswith("#"))
        print(f"Found {cnt} URNs in {f}")

    resp = input("Proceed with download? (yes/no): ").strip().lower()
    if resp != "yes":
        print("Cancelled.")
        return

    dl = NASAInSightDownloader()
    dl.run()

if __name__ == "__main__":
    main()