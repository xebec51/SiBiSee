from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class HoldoutSession:
    participant_id: str
    session_id: str
    device_label: str
    background: str
    lighting: str
    distance: str
    notes: str = ""


def validate_session(session: HoldoutSession) -> None:
    forbidden = {"", "name", "email", "phone", "address"}
    for field, value in asdict(session).items():
        normalized = value.strip().lower()
        if field != "notes" and normalized in forbidden:
            raise ValueError(f"{field} harus diisi dengan ID/kategori non-pribadi.")


def metadata_path(output_dir: Path) -> Path:
    return output_dir / "metadata.csv"


def append_metadata(output_dir: Path, rows: list[dict[str, str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_path(output_dir)
    fieldnames = [
        "relative_path",
        "class_name",
        "participant_id",
        "session_id",
        "device_label",
        "background",
        "lighting",
        "distance",
        "captured_at_unix",
        "notes",
    ]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def write_session_summary(output_dir: Path, session: HoldoutSession, class_names: list[str], dry_run: bool) -> None:
    payload = {
        "session": asdict(session),
        "class_names": class_names,
        "dry_run": dry_run,
        "privacy_note": "Participant/session IDs must be pseudonymous; do not store names, email, phone, or address.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "session_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def collect_holdout(
    output_dir: Path,
    class_names: list[str],
    session: HoldoutSession,
    camera_index: int = 0,
    dry_run: bool = False,
) -> Path:
    validate_session(session)
    if not class_names:
        raise ValueError("Minimal satu class harus diberikan.")
    write_session_summary(output_dir, session, class_names, dry_run)
    if dry_run:
        append_metadata(output_dir, [])
        return output_dir

    import cv2

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Kamera index {camera_index} tidak dapat dibuka.")

    try:
        for class_name in class_names:
            class_dir = output_dir / "images" / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            print(f"Class: {class_name}. Tekan SPACE untuk capture, n untuk class berikutnya, q untuk selesai.")
            captured_rows: list[dict[str, str]] = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Frame kamera gagal dibaca.")
                cv2.imshow("SiBiSee holdout capture", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    timestamp = int(time.time() * 1000)
                    filename = f"{session.session_id}_{session.participant_id}_{class_name}_{timestamp}.jpg"
                    image_path = class_dir / filename
                    cv2.imwrite(str(image_path), frame)
                    captured_rows.append(
                        {
                            "relative_path": str(image_path.relative_to(output_dir)),
                            "class_name": class_name,
                            "participant_id": session.participant_id,
                            "session_id": session.session_id,
                            "device_label": session.device_label,
                            "background": session.background,
                            "lighting": session.lighting,
                            "distance": session.distance,
                            "captured_at_unix": str(timestamp),
                            "notes": session.notes,
                        }
                    )
                elif key == ord("n"):
                    append_metadata(output_dir, captured_rows)
                    break
                elif key == ord("q"):
                    append_metadata(output_dir, captured_rows)
                    return output_dir
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect pseudonymous real-world holdout images for SiBiSee.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name", action="append", dest="class_names", required=True)
    parser.add_argument("--participant-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--device-label", required=True)
    parser.add_argument("--background", required=True)
    parser.add_argument("--lighting", required=True)
    parser.add_argument("--distance", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    session = HoldoutSession(
        participant_id=args.participant_id,
        session_id=args.session_id,
        device_label=args.device_label,
        background=args.background,
        lighting=args.lighting,
        distance=args.distance,
        notes=args.notes,
    )
    output_dir = collect_holdout(args.output_dir, args.class_names, session, args.camera_index, args.dry_run)
    print(f"holdout_dir: {output_dir}")


if __name__ == "__main__":
    main()
