from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def upload(url: str, image: bytes) -> tuple[int, object]:
    boundary = uuid.uuid4().hex
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="bird.jpg"\r\n'
            "Content-Type: image/jpeg\r\n\r\n"
        ).encode()
        + image
        + f"\r\n--{boundary}--\r\n".encode()
    )
    request = urllib.request.Request(
        url + "/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    for endpoint in ("health", "ready"):
        with urllib.request.urlopen(args.url + "/" + endpoint, timeout=5) as response:
            assert response.status == 200
            print(f"{endpoint}: 200")
    status, predictions = upload(args.url, args.image.read_bytes())
    assert status == 200 and isinstance(predictions, list) and len(predictions) <= 5
    print("predict:", status, json.dumps(predictions))
    status, _ = upload(args.url, b"not an image")
    assert status == 400
    print("invalid image:", status)
