# Object Tracking with Trackline
This project utilizes YOLOv9 and DeepSORT for object tracking, combined with a line-drawing feature to visualize the path of tracked objects.

## Reproductbility

### 1. Clone this project

```bash
git clone https://github.com/quzanh1130/Object_tracking_trackline.git
cd Object_tracking_trackline
```

### 2. Install package

```bash
pip install -r setup.txt
```

### 3. Run this project

- `--weights`: model path or triton URL
- `--source`: file/dir/URL/glob/screen/0(webcam)
- `--device`: cuda device, i.e. 0 or 0,1,2,3 or cpu
- `--mode`: mode show line or not. 0: not show, 1: show line

``` bash
  python object_tracking.py --weights weight\yolov9-c-converted.pt --source data_ext/test.mp4 --device cuda --mode 1
```

