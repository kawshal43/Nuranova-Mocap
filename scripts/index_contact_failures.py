"""Map Blender contact findings back to synchronized source video PTS/frame indices."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine.data_format.motion_io import load_motion_file
from multi_video_studio.package_ingest import ingest_multi_video_source
from multi_video_studio.review_state import ReviewState
from multi_video_studio.review_video import ReviewVideoReader


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion",type=Path)
    parser.add_argument("contacts",type=Path)
    parser.add_argument("videos",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    doc=load_motion_file(args.motion)
    state=ReviewState(clocks=doc.sync_info)
    package=ingest_multi_video_source(args.videos)
    indices={}
    for uid,clip in package.motion_clips.items():
        reader=ReviewVideoReader(clip.path)
        indices[uid]=reader.presentation_times.copy()
        reader.close()
    report=json.loads(args.contacts.read_text(encoding="utf-8"))
    samples=report.get("validation",report.get("before",[]))
    fields=["frame","global_s","pair","kind","depth_mm","triangle_crossings","intent_review"]
    fields += [uid+suffix for uid in indices for suffix in ("_file","_frame","_pts_s")]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",encoding="utf-8-sig",newline="") as output:
        writer=csv.DictWriter(output,fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            global_s=float(doc.frame_timestamps_ns[0])*1e-9+(sample["frame"]-1)/doc.fps
            sources={}
            for uid,times in indices.items():
                i=int(np.argmin(abs(times-state.local_time(uid,global_s))))
                sources.update({uid+"_file":package.motion_clips[uid].path.name,
                    uid+"_frame":i,uid+"_pts_s":round(float(times[i]),6)})
            for contact in sample["contacts"]:
                writer.writerow({"frame":sample["frame"],"global_s":round(global_s,6),
                    "pair":contact["a"]+" / "+contact["b"],"kind":contact["kind"],
                    "depth_mm":round(contact["depth_m"]*1000,3),
                    "triangle_crossings":contact["triangle_crossings"],"intent_review":"unreviewed",**sources})
    print(args.output)


if __name__=="__main__":
    main()
