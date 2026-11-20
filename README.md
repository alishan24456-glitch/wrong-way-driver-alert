# Wrong-Way Driver Alert

A divided-highway direction check: an elevated camera watches both
carriageways, YOLOv8n detects every vehicle, a centroid tracker
follows each one, and the system measures its direction of travel from
the centroid movement (a least-squares velocity fit over the last
~1.5 s). Each carriageway has its own legal direction, measured on the
demo clip — the road curves, so the two directions are not simply
opposite:

| Carriageway | Legal direction |
|---|---|
| Left (near) | −20° (up-right, away from the camera) |
| Right (far) | +42° (down-right, toward the camera) |

Direction is judged by the dot product of the vehicle's unit velocity
with the legal unit vector: above +0.5 is **LEGAL**, below −0.5 is
**WRONG WAY**, in between is **UNCERTAIN**. A vehicle needs 1 s of
tracking and 25 px/s of speed before it is judged at all — a parked or
barely-moving vehicle is never flagged. WRONG WAY raises a red banner,
a per-vehicle direction arrow and a timestamped event log; the board
shows vehicles tracked, legal / wrong-way counts and wrong-way events.

## How to run

```bash
pip install -r requirements.txt
python wrong_way_driver_alert.py                 # opens a file picker
python wrong_way_driver_alert.py --source road.mp4
python wrong_way_driver_alert.py --source 0      # webcam
python wrong_way_driver_alert.py --simulate      # rendered highway
```

Weights: `yolov8n.onnx` (12.8 MB) ships in this folder. If it is
missing, the script exports it from `yolov8n.pt` with Ultralytics
(downloading the `.pt` if needed) and otherwise exits with
instructions. Press Q to quit. The video is written to `output.mp4`,
the timestamped event log to `output_events.log` and the final counters
to `output_summary.txt` next to it.

To use the tool on a different camera angle, pass the legal directions
for that scene: `--legal-left` and `--legal-right` (degrees; 0 = right,
90 = down, image coordinates). The central-reservation split line
(`SPLIT_X0`, `SPLIT_SLOPE` at the top of the script) may also need
adjusting for a different viewpoint.

## Verified results

**Real footage (the delivered demo)** — first 30 s of a fresh
free-license Pexels clip of an elevated view over a divided motorway
(video 12420164), rendered at 1280×720, 30 fps:

- 2,197 vehicle detections, 67 vehicles tracked, **44 judged LEGAL**
- **0 WRONG WAY events**, 0.0 s with a wrong-way vehicle, final
  alert CLEAR — as expected: every vehicle on the real clip is
  travelling legally
- Direction angles measured on the real clip match the legal
  directions: left-carriageway vehicles around −7° to −14°,
  right-carriageway vehicles around +18° to +41°

**Simulation (internal logic check only — not the demo)** — a rendered
divided highway where one car drives down the left carriageway against
the traffic:

- 20 vehicles judged LEGAL, **1 WRONG WAY event** (vehicle #7,
  travelling 160° against the left carriageway's legal −20°),
  first flagged at 9.0 s (1 s after it entered, as required by the
  1 s tracking rule), 9.2 s with a wrong-way vehicle, final alert
  CLEAR after it left the frame — no legal vehicle was ever flagged

## Honest limits

- Direction is judged against a **documented legal direction per
  carriageway for the scene**; the system does not know the road's real
  signage, and the split line is a straight-line approximation of the
  central reservation. A vehicle driving close to the reservation can
  be assigned to the wrong band and read UNCERTAIN — it is never
  flagged on a guess.
- A vehicle that is turning, changing lanes across the reservation or
  crawling in congestion can read UNCERTAIN or TOO SLOW.
- Far, small vehicles move slowly in pixels, so many stay below the
  25 px/s speed gate and are never judged; on the real demo 23 of 67
  tracked vehicles never reached a verdict (mostly short-lived or
  far-away tracks).
- The demo clip contains no wrong-way vehicle, so the alert itself is
  verified in simulation only — see above.
- YOLOv8n occasionally fires a transient false box on road markings;
  such boxes are tracked like real vehicles but are judged by the same
  direction rules and never raised an alert on the demo.

## Demo note

The demo video (`wrong-way-driver-alert-demo.mp4`) is this code running
on the real Pexels clip above — not a simulation. The simulation render
is an internal logic check only and is not presented as a demo.

## Files

- `wrong_way_driver_alert.py` — the whole tool (real mode + simulation)
- `yolov8n.onnx` — YOLOv8n weights (12.8 MB)
- `requirements.txt`, `.gitignore`

Author: Abdul Hanan — Computer Vision & AI Developer
