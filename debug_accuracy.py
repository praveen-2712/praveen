import sys, os, cv2, numpy as np, torch
sys.path.append('src')
import timm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=4)
sd = torch.load('weights/tumor_classifier.pth', map_location=device, weights_only=False)
sd = {k.replace('backbone.', ''): v for k, v in sd.items()}
model.load_state_dict(sd, strict=False)
model.to(device).eval()

label_map = {0: 'glioma', 1: 'meningioma', 2: 'no_tumor', 3: 'pituitary'}
norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
norm_std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

def classify_clahe(path):
    img_bgr = cv2.imread(path)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(img_gray, (380, 380))
    channels = []
    for clip in [1.5, 2.0, 2.5]:
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
        channels.append(clahe.apply(gray_resized))
    stack = np.stack(channels, axis=-1)
    t = torch.from_numpy(stack).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0
    t = (t - norm_mean) / norm_std
    probs_list = []
    with torch.no_grad():
        for _ in range(15):
            out = model(t)
            probs_list.append(torch.softmax(out,dim=1).cpu().numpy()[0])
    return np.mean(probs_list, axis=0)

total_correct = 0
total_samples = 0
for cls in ['glioma', 'meningioma', 'notumor', 'pituitary']:
    cls_dir = os.path.join('data/classification/Testing', cls)
    if not os.path.exists(cls_dir):
        cls_dir = os.path.join('data/classification/Testing', cls.replace('notumor', 'no_tumor'))
    if not os.path.exists(cls_dir):
        print(f'SKIP {cls} - dir not found')
        continue
    files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    correct = 0
    pred_counts = {v: 0 for v in label_map.values()}
    for f in files:
        mean_p = classify_clahe(os.path.join(cls_dir, f))
        pred = label_map[int(np.argmax(mean_p))]
        pred_counts[pred] += 1
        if pred.replace('_','') == cls.replace('_',''):
            correct += 1
    total_correct += correct
    total_samples += len(files)
    print(f'{cls}: {correct}/{len(files)} ({round(correct/len(files)*100,1)}%) | Predicted as: {pred_counts}')

print(f'\nOverall: {total_correct}/{total_samples} ({round(total_correct/total_samples*100,2)}%)')
