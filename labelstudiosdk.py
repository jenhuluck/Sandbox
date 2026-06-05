import argparse
import json
import os
from urllib.parse import unquote

import requests


def extract_filename(task_image_value: str) -> str:
    
    decoded = unquote(task_image_value)
    decoded = decoded.replace("\\", "/")
    return os.path.basename(decoded)


def get_project_tasks(ls_url, project_id, headers):
    tasks = []
    page = 1

    while True:
        url = f"{ls_url}/api/projects/{project_id}/tasks"
        params = {
            "page": page,
            "page_size": 100,
        }

        r = requests.get(url, headers=headers, params=params)
        r.raise_for_status()
        data = r.json()

        if isinstance(data, list):
            batch = data
        else:
            batch = data.get("tasks") or data.get("results") or []

        if not batch:
            break

        tasks.extend(batch)
        page += 1

    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--pred-json", required=True)
    parser.add_argument("--model-version", default="coco_preload")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    headers = {
        "Authorization": f"Token {args.api_key}",
        "Content-Type": "application/json",
    }

    with open(args.pred_json, "r", encoding="utf-8") as f:
        preds_by_filename = json.load(f)

    # Normalize prediction keys to basename only
    preds_by_filename = {
        os.path.basename(k.replace("\\", "/")): v
        for k, v in preds_by_filename.items()
    }

    tasks = get_project_tasks(args.url, args.project_id, headers)
    print(f"Found {len(tasks)} tasks in project {args.project_id}")
    print(f"Found {len(preds_by_filename)} filenames in prediction JSON")

    matched = 0
    uploaded = 0
    missing_prediction = 0
    failed = 0

    for task in tasks:
        task_id = task["id"]
        image_value = task["data"]["image"]
        filename = extract_filename(image_value)

        if filename not in preds_by_filename:
            missing_prediction += 1
            continue

        result = preds_by_filename[filename]
        matched += 1

        if not result:
            continue

        payload = {
            "task": task_id,
            "model_version": args.model_version,
            "result": result,
        }

        if args.dry_run:
            print(f"[DRY RUN] Would upload {len(result)} boxes to task {task_id}: {filename}")
            continue

        r = requests.post(
            f"{args.url}/api/predictions/",
            headers=headers,
            json=payload,
        )

        if r.status_code in (200, 201):
            uploaded += 1
        else:
            failed += 1
            print(f"Failed task {task_id}, {filename}: {r.status_code} {r.text}")

    print("Done.")
    print(f"Matched tasks: {matched}")
    print(f"Uploaded predictions: {uploaded}")
    print(f"Tasks without matching prediction: {missing_prediction}")
    print(f"Failed uploads: {failed}")


if __name__ == "__main__":
    main()