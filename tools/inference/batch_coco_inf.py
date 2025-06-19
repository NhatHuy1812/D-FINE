import os, glob, json, argparse
from PIL import Image, ImageDraw, ImageFont   # ← add ImageDraw, ImageFont
import torch, torch.nn as nn
import torchvision.transforms as T
import time                                    # <<<< added

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
    latencies = []

    # make output‐folder for drawn images
    draw_dir = os.path.join(args.input_dir, "../drawn_finetuned")
    os.makedirs(draw_dir, exist_ok=True)

    for idx, img_path in enumerate(files):
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)                # ← prepare to draw
        w, h = img.size

        inp   = tf(img).unsqueeze(0).to(device)
        sizes = torch.tensor([[w,h]], device=device)

        t0 = time.time()                      # <<<< added
        with torch.no_grad():
            labels, boxes, scores = model(inp, sizes)
        t1 = time.time()                      # <<<< added
        latencies.append(t1 - t0)             # <<<< added

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

    with open(args.output, "w") as f:
        json.dump(results, f)

    total_time = sum(latencies)               # <<<< added
    n = len(latencies)                        # <<<< added
    fps = n / total_time if total_time>0 else 0  # <<<< added
    avg_lat = total_time / n if n>0 else 0      # <<<< added

    print(f"Saved {len(results)} detections to {args.output}")
    print(f"Processed {n} images in {total_time:.3f}s — FPS: {fps:.2f}, avg latency: {avg_lat:.3f}s")  # <<<< added

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