"""Read public release metadata without downloading dataset archives."""

import argparse
import http.cookiejar
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from sdgm.io import save_json


def public_share_listing(url, password):
    cookie = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie))
    headers = {"User-Agent": "Mozilla/5.0", "Referer": url}
    page = (
        opener.open(urllib.request.Request(url, headers=headers), timeout=20)
        .read()
        .decode()
    )
    shareid = re.search(r'shareid[:=]\\?"?(\d+)', page).group(1)
    owner = re.search(r'share_uk[:=]\\?"?(\d+)', page).group(1)
    suffix = urllib.parse.urlparse(url).path.rsplit("/", 1)[-1][1:]
    verification = urllib.request.Request(
        "https://pan.baidu.com/share/verify?surl=" + suffix,
        data=urllib.parse.urlencode({"pwd": password}).encode(),
        headers=headers,
    )
    result = json.load(opener.open(verification, timeout=20))
    if result.get("errno") != 0:
        raise RuntimeError(f"Public share verification returned {result.get('errno')}")
    files = []

    def visit(directory=None, depth=0):
        params = dict(
            uk=owner, shareid=shareid, page=1, num=100, order="name", desc=0, web=1
        )
        if directory:
            params["dir"] = directory
        else:
            params["root"] = 1
        request = urllib.request.Request(
            "https://pan.baidu.com/share/list?" + urllib.parse.urlencode(params),
            headers=headers,
        )
        result = json.load(opener.open(request, timeout=20))
        if result.get("errno") != 0:
            raise RuntimeError(f"List error {result.get('errno')}")
        for item in result.get("list", []):
            row = {
                key: item.get(key)
                for key in ["server_filename", "path", "size", "isdir", "server_mtime"]
            }
            opaque = item.get("md5", "")
            row["provider_md5_field"] = opaque
            row["provider_md5_is_hex_digest"] = bool(
                re.fullmatch("[0-9a-fA-F]{32}", opaque)
            )
            files.append(row)
            if item.get("isdir") and depth < 3:
                visit(item["path"], depth + 1)

    visit()
    return dict(
        url=url,
        shareid=shareid,
        public_share_owner_id=owner,
        files=files,
        caution="Provider fields containing nonhex characters are opaque identifiers, not verified MD5 digests",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/provenance_20261125")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    urls = {
        "gdros_official_readme": "https://raw.githubusercontent.com/Zi-Xuan-Sun/GDROS/main/README.md",
        "gdros_paper": "https://arxiv.org/html/2511.00598v1",
        "soma_paper": "https://arxiv.org/html/2511.13168v1",
        "sopatch_secondary_readme": "https://raw.githubusercontent.com/Bend1031/DH-net/master/README.md",
    }
    records = []
    for name, url in urls.items():
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=20) as response:
                data = response.read().decode()
            text = (
                " ".join(re.sub(r"<[^>]+>", " ", data).split())
                if "paper" in name
                else data
            )
            if "paper" in name:
                excerpts = [
                    text[max(0, m.start() - 150) : m.start() + 800]
                    for m in re.finditer(
                        r"SOPatch|5,600|precisely registered|finely registered|SEN1-2|geometrical transformations",
                        text,
                    )
                ]
                save_json(
                    output / f"{name}_excerpts.json", dict(url=url, excerpts=excerpts)
                )
            else:
                (output / f"{name}.md").write_text(data)
            records.append(
                dict(
                    name=name,
                    url=url,
                    status="retrieved",
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
            )
        except Exception as error:
            records.append(
                dict(name=name, url=url, status="unavailable", error=str(error))
            )
    for name, url, password in [
        (
            "gdros_public_share",
            "https://pan.baidu.com/s/17dNYVhNObbVjgcY2SmlM-w",
            "ukpw",
        ),
        (
            "sopatch_secondary_share",
            "https://pan.baidu.com/s/12yIheGOg6JTTsYAQfX7Pfg",
            "1234",
        ),
    ]:
        try:
            result = public_share_listing(url, password)
            save_json(output / f"{name}.json", result)
            records.append(
                dict(
                    name=name,
                    url=url,
                    status="listed",
                    files=len(result["files"]),
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
            )
        except Exception as error:
            records.append(
                dict(name=name, url=url, status="unavailable", error=str(error))
            )
    save_json(output / "source_checks.json", records)
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
