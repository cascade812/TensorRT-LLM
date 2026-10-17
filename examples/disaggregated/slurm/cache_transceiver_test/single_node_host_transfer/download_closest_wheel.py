#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Download the closest compatible post-merge wheel in the current Git history."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import os
import platform
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

DEFAULT_ARTIFACT_URL = (
    "https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/LLM/main/L0_PostMerge/"
)
DEFAULT_OUTPUT_DIR = Path("build")
BUILD_INFO_NAME = "build_info.txt"
DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024
PROGRESS_INTERVAL = 128 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 60
METADATA_WORKERS = 8


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for key, value in attrs:
            if key == "href" and value is not None:
                self.links.append(value)
                return


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "TensorRT-LLM-wheel-downloader"})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return response.read()


def _list_links(url: str) -> list[str]:
    parser = _LinkParser()
    parser.feed(_fetch(url).decode("utf-8", "replace"))
    return parser.links


def _parse_build_numbers(links: list[str]) -> list[int]:
    builds = []
    for link in links:
        path = urllib.parse.urlsplit(link).path.rstrip("/")
        name = path.rsplit("/", 1)[-1]
        if name.isdigit():
            builds.append(int(name))
    return sorted(set(builds), reverse=True)


def _parse_commit(build_info: bytes) -> str | None:
    for line in build_info.decode("utf-8", "replace").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "commit" and value.strip():
            return value.strip().lower()
    return None


def _fetch_build_commit(base_url: str, build: int) -> tuple[int, str | None]:
    url = urllib.parse.urljoin(base_url, f"{build}/{BUILD_INFO_NAME}")
    try:
        return build, _parse_commit(_fetch(url))
    except OSError:
        return build, None


def _build_commits(base_url: str, builds: list[int]) -> dict[int, str]:
    commits = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=METADATA_WORKERS) as executor:
        futures = [executor.submit(_fetch_build_commit, base_url, build) for build in builds]
        for future in concurrent.futures.as_completed(futures):
            build, commit = future.result()
            if commit is not None:
                commits[build] = commit
    return commits


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", os.fspath(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def _candidate_builds(repo: Path, build_commits: dict[int, str]) -> list[tuple[int, str, int]]:
    history = set(_git(repo, "rev-list", "HEAD").lower().splitlines())
    matching_commits = {commit for commit in build_commits.values() if commit in history}
    distances = {
        commit: int(_git(repo, "rev-list", "--count", "--ancestry-path", f"{commit}..HEAD"))
        for commit in matching_commits
    }
    candidates = [
        (build, commit, distances[commit])
        for build, commit in build_commits.items()
        if commit in distances
    ]
    return sorted(candidates, key=lambda candidate: (candidate[2], -candidate[0]))


def _architecture_directory(machine: str) -> str:
    aliases = {
        "aarch64": "aarch64-linux-gnu",
        "arm64": "aarch64-linux-gnu",
        "aarch64-linux-gnu": "aarch64-linux-gnu",
        "x86_64": "x86_64-linux-gnu",
        "amd64": "x86_64-linux-gnu",
        "x86_64-linux-gnu": "x86_64-linux-gnu",
    }
    architecture = aliases.get(machine.lower())
    if architecture is None:
        supported = ", ".join(sorted(aliases))
        raise ValueError(f"unsupported architecture {machine!r}; expected one of: {supported}")
    return architecture


def _default_python_tag() -> str:
    if sys.implementation.name != "cpython":
        raise RuntimeError(
            "automatic wheel selection requires CPython; pass --python-tag explicitly"
        )
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _installed_torch_version() -> str | None:
    try:
        return importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None


def _ngc_pytorch_release(torch_version: str) -> str | None:
    match = re.search(r"(?:^|[.+])nv(?P<year>\d{2})\.(?P<month>\d{1,2})(?:[.+]|$)", torch_version)
    if match is None:
        return None

    month = int(match.group("month"))
    if not 1 <= month <= 12:
        return None
    return f"{match.group('year')}{month:02d}"


def _resolve_ngc_variant(
    ngc_override: bool | None, torch_version: str | None
) -> tuple[bool, str | None]:
    ngc_pytorch_release = _ngc_pytorch_release(torch_version) if torch_version is not None else None
    ngc = ngc_override if ngc_override is not None else ngc_pytorch_release is not None
    return ngc, ngc_pytorch_release if ngc else None


def _select_wheel(
    links: list[str],
    python_tag: str,
    ngc: bool,
    ngc_pytorch_release: str | None = None,
) -> str | None:
    tag_pattern = re.compile(rf"-{re.escape(python_tag)}-{re.escape(python_tag)}-")
    matches = []
    for link in links:
        name = Path(urllib.parse.urlsplit(link).path).name
        if name.endswith(".whl") and tag_pattern.search(name):
            ngc_match = re.search(r"\+ngcpytorch(?P<release>\d+)", name)
            is_ngc = ngc_match is not None
            if is_ngc == ngc:
                if ngc and ngc_pytorch_release is not None:
                    release = ngc_match.group("release") if ngc_match is not None else None
                    if release != ngc_pytorch_release:
                        continue
                matches.append(link)
    if not matches:
        return None
    return min(matches, key=lambda link: Path(urllib.parse.urlsplit(link).path).name)


def _find_closest_wheel(
    base_url: str,
    repo: Path,
    architecture: str,
    python_tag: str,
    ngc: bool,
    ngc_pytorch_release: str | None = None,
) -> tuple[int, str, int, str]:
    base_url = f"{base_url.rstrip('/')}/"
    builds = _parse_build_numbers(_list_links(base_url))
    if not builds:
        raise RuntimeError(f"no build directories found at {base_url}")

    build_commits = _build_commits(base_url, builds)
    candidates = _candidate_builds(repo, build_commits)
    if not candidates:
        raise RuntimeError(
            f"none of the {len(build_commits)} builds with commit metadata is in HEAD's history"
        )

    for build, commit, distance in candidates:
        platform_url = urllib.parse.urljoin(base_url, f"{build}/{architecture}/")
        try:
            wheel_link = _select_wheel(
                _list_links(platform_url), python_tag, ngc, ngc_pytorch_release
            )
        except OSError:
            continue
        if wheel_link is not None:
            return build, commit, distance, urllib.parse.urljoin(platform_url, wheel_link)

    if ngc and ngc_pytorch_release is not None:
        variant = f"NGC PyTorch {ngc_pytorch_release}"
    else:
        variant = "NGC PyTorch" if ngc else "standard"
    raise RuntimeError(
        f"no {variant} {python_tag} wheel for {architecture} was found on a build in HEAD's history"
    )


def _format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def _print_progress(downloaded: int, total: int | None) -> None:
    if total is None:
        message = f"Downloaded {_format_size(downloaded)}"
    else:
        percentage = downloaded * 100.0 / total
        message = (
            f"Downloaded {_format_size(downloaded)} / {_format_size(total)} ({percentage:.1f}%)"
        )
    print(message, file=sys.stderr)


def _download(url: str, output_dir: Path, force: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urllib.parse.urlsplit(url).path).name
    destination = output_dir / filename
    if destination.exists() and not force:
        print(f"Wheel already exists: {destination}")
        return destination

    temporary = destination.with_name(f".{destination.name}.part")
    request = urllib.request.Request(url, headers={"User-Agent": "TensorRT-LLM-wheel-downloader"})
    digest = hashlib.sha256()
    downloaded = 0
    next_progress = PROGRESS_INTERVAL
    try:
        with (
            urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response,
            temporary.open("wb") as output,
        ):
            content_length = response.headers.get("Content-Length")
            total = int(content_length) if content_length is not None else None
            expected_digest = response.headers.get("X-Checksum-Sha256")
            while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                output.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                if downloaded >= next_progress:
                    _print_progress(downloaded, total)
                    next_progress += PROGRESS_INTERVAL
    except OSError:
        temporary.unlink(missing_ok=True)
        raise

    if total is not None and downloaded != total:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded {downloaded} bytes, expected {total} bytes")
    if expected_digest is not None and digest.hexdigest().lower() != expected_digest.lower():
        temporary.unlink(missing_ok=True)
        raise RuntimeError("downloaded wheel does not match Artifactory's SHA-256 checksum")

    os.replace(temporary, destination)
    _print_progress(downloaded, total)
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the closest compatible TensorRT-LLM post-merge wheel whose build commit "
            "is in the current branch's history. The wheel variant is detected from the "
            "installed PyTorch unless overridden."
        )
    )
    parser.add_argument(
        "--artifact-url",
        default=DEFAULT_ARTIFACT_URL,
        help="L0_PostMerge Artifactory directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="download directory (default: %(default)s)",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Git worktree whose HEAD should be matched (default: current directory)",
    )
    parser.add_argument(
        "--architecture",
        default=platform.machine(),
        help="target machine or GNU platform triple (default: detected host architecture)",
    )
    parser.add_argument(
        "--python-tag",
        default=None,
        help="CPython wheel tag such as cp310 or cp312 (default: running Python)",
    )
    variant_group = parser.add_mutually_exclusive_group()
    variant_group.add_argument(
        "--ngc",
        dest="ngc",
        action="store_const",
        const=True,
        help="force selection of an +ngcpytorch wheel",
    )
    variant_group.add_argument(
        "--standard",
        dest="ngc",
        action="store_const",
        const=False,
        help="force selection of a standard wheel",
    )
    parser.set_defaults(ngc=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the selected wheel without downloading it",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing wheel with the same filename",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    architecture = _architecture_directory(args.architecture)
    python_tag = args.python_tag or _default_python_tag()
    if re.fullmatch(r"cp\d+", python_tag) is None:
        raise ValueError(f"invalid Python tag {python_tag!r}; expected a value such as cp312")

    torch_version = _installed_torch_version()
    ngc, ngc_pytorch_release = _resolve_ngc_variant(args.ngc, torch_version)
    if ngc:
        if ngc_pytorch_release is None:
            print("PyTorch wheel variant: NGC PyTorch (forced)")
        else:
            print(
                f"PyTorch wheel variant: NGC PyTorch {ngc_pytorch_release} "
                f"(detected from torch {torch_version})"
            )
    else:
        if args.ngc is False:
            reason = "forced"
        elif torch_version is None:
            reason = "torch is not installed"
        else:
            reason = f"detected from torch {torch_version}"
        print(f"PyTorch wheel variant: standard ({reason})")

    build, commit, distance, wheel_url = _find_closest_wheel(
        args.artifact_url,
        args.repo,
        architecture,
        python_tag,
        ngc,
        ngc_pytorch_release if ngc else None,
    )
    print(f"Selected build: {build}")
    print(f"Build commit: {commit}")
    print(f"Distance from HEAD: {distance} commit(s)")
    print(f"Wheel URL: {wheel_url}")
    if args.dry_run:
        return 0

    destination = _download(wheel_url, args.output_dir, args.force)
    print(f"Downloaded wheel: {destination}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
