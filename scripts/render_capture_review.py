"""Render timestamp-aligned source cameras beside the solved rig for visual validation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import av
import cv2
import numpy as np
from PySide6.QtGui import QFont, QImage
from PySide6.QtWidgets import QApplication

from desktop.ui.widgets.viewport3d import Canvas3DWidget
from engine.data_format.motion_io import load_motion_file


class TimedReader:
    def __init__(self, path, start):
        self.container = av.open(str(path))
        stream = self.container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 2
        self.container.seek(int(max(0, start - 1) * av.time_base), backward=True)
        self.frames = self.container.decode(stream)
        self.previous = self.next = None

    def at(self, time_s):
        while self.next is None or float(self.next.pts * self.next.time_base) < time_s:
            self.previous = self.next
            try:
                self.next = next(self.frames)
            except StopIteration:
                self.next = self.previous
                break
        frame = self.next
        if self.previous and abs(float(self.previous.pts * self.previous.time_base) - time_s) < abs(
            float(frame.pts * frame.time_base) - time_s
        ):
            frame = self.previous
        return frame.reformat(width=480, height=270, format="bgr24").to_ndarray()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path)
    parser.add_argument("videos", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, default=10)
    args = parser.parse_args()
    doc = load_motion_file(args.motion)
    track = doc.corrected_motion or doc.raw_motion
    app = QApplication.instance() or QApplication([])
    app.setFont(QFont("Segoe UI", 10))
    canvas = Canvas3DWidget()
    canvas.setMinimumSize(480, 270)
    canvas.resize(480, 270)
    canvas.set_display_mode("both")
    canvas.show_cameras = False
    canvas.distance = 2.8
    canvas.target = np.array([*np.median(track.root_translation[:, :2], axis=0), 0.92])
    cameras = doc.cameras[:3]
    readers = []
    first_global = doc.frame_timestamps_ns[0] * 1e-9

    def local_time(camera, global_time):
        estimate = doc.sync_info.get(camera["camera_uid"], {})
        slope = estimate.get("drift_slope", 1)
        offset = estimate.get("offset_s", 0)
        pivot = estimate.get("drift_pivot_s", 0)
        origin = estimate.get("audio_start_s", 0)
        return (global_time - offset + (slope - 1) * pivot) / slope + origin

    for camera in cameras:
        readers.append(
            TimedReader(args.videos / camera["source"], local_time(camera, first_global))
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (960, 540)
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create review video.")
    stride = max(1, int(round(doc.fps / args.fps)))
    frame_indices = list(range(0, doc.num_frames, stride))
    sample_indices = {
        frame_indices[i] for i in np.linspace(0, len(frame_indices) - 1, 8).astype(int)
    }
    snapshots = []
    try:
        for index in frame_indices:
            time_s = doc.frame_timestamps_ns[index] * 1e-9
            tiles = []
            for camera, reader in zip(cameras, readers):
                tile = reader.at(local_time(camera, time_s))
                cv2.putText(
                    tile,
                    camera["camera_uid"] + f"  {local_time(camera, time_s):.2f}s",
                    (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    1,
                )
                tiles.append(tile)
            while len(tiles) < 3:
                tiles.append(np.zeros((270, 480, 3), np.uint8))
            canvas.set_scene_frame(
                track.joint_positions[index],
                track.joint_confidence[index],
                global_matrices=track.global_matrices[index],
            )
            app.processEvents()
            qimage = canvas.grab().toImage().convertToFormat(QImage.Format.Format_RGB888)
            array = (
                np.frombuffer(qimage.bits(), np.uint8)
                .reshape(qimage.height(), qimage.bytesPerLine())[:, : qimage.width() * 3]
                .reshape(qimage.height(), qimage.width(), 3)
            )
            rig_image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            cv2.putText(
                rig_image,
                f"Solved frame {index + 1} / {doc.num_frames}",
                (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            combined = np.vstack([np.hstack(tiles[:2]), np.hstack([tiles[2], rig_image])])
            writer.write(combined)
            if index in sample_indices:
                snapshots.append(cv2.resize(combined, (480, 270)))
            if index % 300 == 0:
                print(f"Review frame {index}/{doc.num_frames}", flush=True)
    finally:
        writer.release()
        for reader in readers:
            reader.container.close()
    if len(snapshots) >= 8:
        sheet = np.vstack([np.hstack(snapshots[:4]), np.hstack(snapshots[-4:])])
        cv2.imwrite(str(args.output.with_suffix(".jpg")), sheet)


if __name__ == "__main__":
    main()
