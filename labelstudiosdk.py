import argparse
import json
import os
from urllib.parse import unquote

from label_studio_sdk import Client


def extract_filename(task_image_value: str) -> str:
    decoded = unquote(task_image_value)
    decoded = decoded.replace("\\", "/")
    return os.path.basename(decoded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--pred-json", required=True)
    parser.add_argument("--model-version", default="coco_preload")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ls = Client(url=args.url, api_key=args.api_key)
    project = ls.get_project(args.project_id)

    print("Connected to project:", project.id, project.title)

    with open(args.pred_json, "r", encoding="utf-8") as f:
        preds_by_filename = json.load(f)

    preds_by_filename = {
        os.path.basename(k.replace("\\", "/")): v
        for k, v in preds_by_filename.items()
    }

    tasks = project.get_tasks()
    print("Tasks found:", len(tasks))
    print("Prediction files:", len(preds_by_filename))

    matched = 0
    uploaded = 0
    missing = 0

    for task in tasks:
        task_id = task["id"]
        image_value = task["data"]["image"]
        filename = extract_filename(image_value)

        if filename not in preds_by_filename:
            missing += 1
            continue

        result = preds_by_filename[filename]
        matched += 1

        if args.dry_run:
            print(f"[DRY RUN] task {task_id}: {filename}, boxes={len(result)}")
            continue

        project.create_prediction(
            task_id=task_id,
            result=result,
            model_version=args.model_version,
        )

        uploaded += 1

    print("Done.")
    print("Matched:", matched)
    print("Uploaded:", uploaded)
    print("Missing predictions:", missing)


if __name__ == "__main__":
    main()