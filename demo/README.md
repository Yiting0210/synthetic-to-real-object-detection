# Demo — model interface and live demo (M3)

Course requirements (due Nov 3): 1) the object detection model is complete; 2) live demonstration: a) single-object image → ID; b) multi-object image → ID and location of every object.

| File | What it does |
|---|---|
| `predict.py` | Command-line interface. `single` uses the M1 classifier (MobileNetV2); `multi` uses the M2 detector (YOLOv8n). Prints JSON with course IDs (e.g. `OBJ011`) and boxes in the input image's pixels. |
| `demo_app.py` | Local web page (Gradio) with one tab per mode, used for the live demo. |
| `requirements.txt` | Packages for the laptop (CPU only). |

## 1. Get the models (once)

Both tools read the repository-level `../models/` folder (change it with `--models-dir`). `classes.json` and `selection.json` are already there. The two weight files are not in git: copy them from Drive into `../models/`:

| File | Source |
|---|---|
| `classes.json` | in git; from `IE7615_project/runs/m1_20261018_022316/data/classes.json` |
| `mobilenetv2.keras` | Drive `IE7615_project/runs/m1_20261018_022316/models/mobilenetv2.keras` |
| `d1_yolov8n_best.pt` | retrained after mask review, cluster run `m2_20261103_225708` |
| `selection.json` | in git; from the retrained run, see `../detection/results/` (confidence threshold chosen on D1-val: 0.55) |

The demo images in `examples/{single,multi}/` come from the same retrained run. Each model goes with the examples of its own run:
the composites were regenerated, so `d1_test_00129.jpg` is a different image for the submitted M2 model (tag `m2-submitted`).

`predict.py` checks that the detector's class names match `classes.json` and stops if they don't.

## 2. Environment (laptop, CPU)

Already set up on the laptop as the conda env `ie7615-demo` (Python 3.11, TensorFlow 2.21, Keras 3.15, PyTorch 2.15 CPU, Ultralytics 8.4.192, Gradio 5.50). The env has `PYTHONNOUSERSITE=1` set, so packages in `AppData\Roaming\Python\Python311` do not leak in; without it, the env silently uses those older copies. To rebuild it elsewhere:

```bash
conda create -n ie7615-demo python=3.11 -y
```

```bash
conda env config vars set PYTHONNOUSERSITE=1 -n ie7615-demo
```

```bash
conda run -n ie7615-demo pip install -r requirements.txt
```

## 3. Run

In PowerShell, inside `demo/` (no activation needed; `conda activate` only works in Anaconda Prompt or after `conda init powershell`):

```powershell
& "E:\CodeTools\anaconda3\envs\ie7615-demo\python.exe" -s demo_app.py
```

In Anaconda Prompt the same works as `conda activate ie7615-demo` followed by `python demo_app.py`.

The demo opens at http://127.0.0.1:7860 after about 15 s (TensorFlow start-up and model warm-up). Command-line use:

```bash
python predict.py single examples/single/OBJ004_0072.jpg
```

```bash
python predict.py multi examples/multi/d1_test_00129.jpg --save annotated.jpg
```

Tested on Oct 29, 2026 (classifier, 8/8 correct on `examples/single`, ~70 ms per image) and on Nov 3, 2026 with the retrained detector (threshold 0.55): all objects identified in 7 of the 8 `examples/multi` composites; ~60 ms per image, first call ~3 s.

## Live demo script (about 3 minutes)

1. **Single object:** click an example (e.g. `OBJ004_0072`, the stop sign) → ID with top-3 probabilities.
2. **Multi-object:** every object is identified in `d1_test_00025`, `00128`, `00129`, `00193`, `00209`, `00227` and `00255`. Point out the boxes and the JSON (ID, confidence, `x0 y0 x1 y1`). Avoid `00026` (the small OBJ001 bottle is missed and one extra OBJ201 box appears), or use it on purpose to show a known limit.
3. **Threshold:** move the slider down to show extra low-confidence boxes, back up to the value chosen on validation.
4. **Outside the training distribution (optional):** a real phone photo of OBJ011 on a cluttered desk. Present it as a case study, not as a measured result (T-real is not built yet).

## Notes

- Test images are used only for showing results; thresholds and models were chosen on validation sets before.
- The M1 classifier carries its own preprocessing (resize with padding to 224×224, RGB 0–255, model scaling inside the model), so `predict.py` repeats exactly the M1 pipeline.
- The detector expects images up to any size; Ultralytics letterboxes them to 640 and maps the boxes back to the original pixels.
