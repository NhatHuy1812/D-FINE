import os
import argparse
import cv2
from pycocotools.coco import COCO
import json
from collections import defaultdict

# derive COCO‐style image_id from filename like 'camera0_M_0001.png'
def get_image_Id(img_name):
    img_name = img_name.split('.png')[0]
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].replace('camera',''))
    sceneIndx  = sceneList.index(img_name.split('_')[1])
    frameIndx  = int(img_name.split('_')[2])
    return int(f"{cameraIndx}{sceneIndx}{frameIndx}")

def visualize_coco(images_dir, ann_file, output_dir, extra_dir=None, display=False):
    os.makedirs(output_dir, exist_ok=True)
    if extra_dir:
        os.makedirs(extra_dir, exist_ok=True)

    # load JSON
    with open(ann_file) as f:
        ann_data = json.load(f)

    # choose branch based on format
    if isinstance(ann_data, dict):
        # full COCO format
        coco = COCO(ann_file)
        cats    = coco.loadCats(coco.getCatIds())
        cat_map = {c['id']: c['name'] for c in cats}
        def iter_images():
            for img_info in coco.loadImgs(coco.getImgIds()):
                ann_ids = coco.getAnnIds(imgIds=img_info['id'])
                anns    = coco.loadAnns(ann_ids)
                yield img_info, anns

    elif isinstance(ann_data, list):
        # detection list format
        dets = ann_data
        # build image_id→filename
        id2file = {}
        for fn in os.listdir(images_dir):
            if fn.lower().endswith(('.png','.jpg','.jpeg')):
                id2file[get_image_Id(fn)] = fn
        # group by image_id
        det_map = defaultdict(list)
        for ann in dets:
            det_map[ann['image_id']].append(ann)
        cat_map = {}  # no name mapping, show ID
        def iter_images():
            for img_id, anns in det_map.items():
                fn = id2file.get(img_id)
                if fn:
                    yield {'id':img_id,'file_name':fn}, anns

    else:
        raise RuntimeError("Unsupported annotation format")

    # common visualization loop
    for img_info, anns in iter_images():
        img_path = os.path.join(images_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ could not read {img_path}")
            continue

        for ann in anns:
            x,y,w,h = map(int, ann['bbox'])
            cid     = ann['category_id']
            cat     = cat_map.get(cid, str(cid))
            score   = ann.get('score', None)
            color   = (0,255,0)

            cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
            label = f"{cat}" + (f":{score:.2f}" if score is not None else "")
            cv2.putText(img, label,
                        (x, max(y-5,0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)

        out_path = os.path.join(output_dir, img_info['file_name'])
        cv2.imwrite(out_path, img)
        if extra_dir:
            cv2.imwrite(os.path.join(extra_dir, img_info['file_name']), img)

        if display:
            cv2.imshow("Viz", img)
            if cv2.waitKey(0)&0xFF==ord('q'):
                break

    if display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", required=True, help="Folder with COCO images")
    p.add_argument("--ann_file",   required=True, help="COCO JSON or detection list")
    p.add_argument("--output_dir", required=True, help="Where to save visualized images")
    p.add_argument("--extra_dir",  default=None, help="Also save to this folder")
    p.add_argument("--display",    action="store_true", help="Show on screen (q to quit)")
    args = p.parse_args()

    visualize_coco(
        args.images_dir,
        args.ann_file,
        args.output_dir,
        extra_dir=args.extra_dir,
        display=args.display
    )