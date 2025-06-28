import os, glob, json, argparse
from PIL import Image, ImageDraw, ImageFont   # ← add ImageDraw, ImageFont
import torch, torch.nn as nn
import torchvision.transforms as T
import time
import cv2                                    # ← use OpenCV for imread
from sklearn.metrics import f1_score          # ← to compute F1‐score

# add project root
import sys
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    ),
)
from src.core import YAMLConfig

def get_image_Id(img_name):
    """Derive COCO-style image_id from filename like 'camera0_M_0001.png'."""
    img_name = img_name.split('.png')[0]
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].split('camera')[1])
    sceneIndx  = sceneList.index(img_name.split('_')[1])
    frameIndx  = int(img_name.split('_')[2])
    # e.g. camera0, scene M, frame 001 → "0M1" → int(“0” + “1” + “1”) 
    imageId = int(f"{cameraIndx}{sceneIndx}{frameIndx}")
    return imageId

def build_model(cfg_path, weights, device):
    cfg = YAMLConfig(cfg_path, resume=weights)
    # disable pretrained backbone
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt = torch.load(weights, map_location="cpu")
    state = ckpt.get("ema", ckpt).get("module", ckpt.get("model", ckpt))
    cfg.model.load_state_dict(state)
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = cfg.model.deploy()
            self.post = cfg.postprocessor.deploy()
        def forward(self, imgs, sizes):
            out = self.backbone(imgs)
            return self.post(out, sizes)
    m = M().to(device).eval()
    return m

def main(args):
    device    = torch.device(args.device)
    model     = build_model(args.config, args.resume, device)
    tf        = T.Compose([T.Resize(args.resize), T.ToTensor()])
    results   = []
    files     = sorted(glob.glob(os.path.join(args.input_dir, "*")))
    sum_time  = 0.0                            # ← accumulator for per‐image inference time
    max_fps   = 25                            # ← upper bound for FPS normalization

    # make output‐folder for drawn images
    draw_dir = os.path.join(args.input_dir, "../drawn_finetuned")
    os.makedirs(draw_dir, exist_ok=True)

    for idx, img_path in enumerate(files):
        # LOAD image via OpenCV
        bgr  = cv2.imread(img_path)
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img  = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)

        w, h = img.size

        inp   = tf(img).unsqueeze(0).to(device)
        sizes = torch.tensor([[w,h]], device=device)

        # measure only inference + post‐process
        t0 = time.time()
        
        with torch.no_grad():
            labels, boxes, scores = model(inp, sizes)
        t1 = time.time()
        sum_time += (t1 - t0)

        img_id = get_image_Id(os.path.basename(img_path))

        lbls = labels[0].cpu().numpy()
        bxs  = boxes[0].cpu().numpy()
        scrs = scores[0].cpu().numpy()
        for l, b, s in zip(lbls, bxs, scrs):
            if s < 0.6:                         # ← only draw if > 0.65
                continue
            x1,y1,x2,y2 = b
            # draw box + label
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1-10), f"{int(l)}:{s:.2f}", fill="red")

            results.append({
                "image_id":    img_id,
                "category_id": int(l),
                "bbox":        [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                "score":       float(s),
            })

        # save the drawn‐on image
        out_path = os.path.join(draw_dir, os.path.basename(img_path))
        img.save(out_path)

    # write COCO results
    with open(args.output, "w") as f:
        json.dump(results, f)

    n_images = len(files)
    fps      = n_images / sum_time if sum_time>0 else 0
    norm_fps = min(fps, max_fps) / max_fps

    # Compute F1-score – implement loading of GT vs preds into y_true, y_pred
    # y_true, y_pred = load_gt_and_preds(...)
    #f1       = f1_score(y_true, y_pred)

    # harmonic mean of norm_fps and f1
    #metric   = (2 * norm_fps * f1) / (norm_fps + f1) if (norm_fps + f1)>0 else 0

    print(f"Saved {len(results)} detections to {args.output}")
    print(f"FPS: {fps:.2f}, norm_FPS: {norm_fps:.2f}")
    #print(f"F1-score: {f1:.3f}, Combined Metric: {metric:.3f}")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-c","--config",    required=True)
    p.add_argument("-r","--resume",    required=True)
    p.add_argument("-i","--input_dir", required=True)
    p.add_argument("-o","--output",    required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--thr",    type=float, default=0.4)
    p.add_argument("--resize", nargs=2, type=int, default=[1280,1280])
    args = p.parse_args()
    main(args)