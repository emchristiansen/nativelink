import pathlib
import re
import sys

import requests

module_bazel_path = pathlib.Path(sys.argv[1])
original = module_bazel_path.read_text()
begin_shas = re.search("# BEGIN SHAS\n", original).end() # pyright: ignore[reportOptionalMemberAccess]
end_shas = re.search("\n        # END SHAS", original).start() # pyright: ignore[reportOptionalMemberAccess]
sha_pattern = re.compile(r"\"(.+\.tar\.xz)\": \"([0-9a-f]+)\"")

results = []
for entry in sha_pattern.finditer(original, begin_shas, end_shas):
    short_url, _ = entry.groups()
    # Rust publishes official checksums alongside each archive. Downloading the
    # archives themselves just to refresh this metadata wastes gigabytes.
    checksum_url = f"https://static.rust-lang.org/dist/{short_url}.sha256"
    print("getting", checksum_url)
    response = requests.get(checksum_url, timeout=30)
    response.raise_for_status()
    checksum = re.fullmatch(r"([0-9a-f]{64})\s+\*?(\S+)\s*", response.text)
    if checksum is None or checksum[2] != pathlib.PurePosixPath(short_url).name:
        raise ValueError(f"Invalid checksum or filename from {checksum_url}")
    results.append(f"        \"{short_url}\": \"{checksum[1]}\",")

if not results:
    raise ValueError(f"No Rust archive entries in {module_bazel_path}")

# Do not replace any metadata until every official checksum has been validated.
revised = original[:begin_shas] + "\n".join(results) + original[end_shas:]
module_bazel_path.write_text(revised)
