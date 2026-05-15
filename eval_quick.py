import sys; sys.path.insert(0,'.')
import torch, yaml, pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
from pathlib import Path
from datasets.ravdess import RAVDESSVideoDataset, IDX_TO_LABEL_6 as IDX_TO_LABEL
from models.video.backbone import VideoEmotionModel

cfg = yaml.safe_load(open('configs/config.yaml'))
tcn = cfg['video']['tcn']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

model = VideoEmotionModel(
    num_classes=cfg['emotions']['num_classes'],
    num_channels=tcn['num_channels'],
    num_levels=tcn['num_levels'],
    kernel_size=tcn['kernel_size'],
    dropout=0.0,
    pretrained=False,
)
model.load_state_dict(torch.load('checkpoints/video_best.pt', map_location='cpu', weights_only=False))
model.eval().to(device)

proc_dir = Path(cfg['data']['processed_dir']) / 'video'

for split in ['val', 'test']:
    df = pd.read_csv(Path(cfg['data']['splits_dir']) / f'{split}.csv')
    df = df[df['ext'] == '.flv'].reset_index(drop=True)
    print(f'{split} samples: {len(df)}')

    ds = RAVDESSVideoDataset(df, proc_dir, window_frames=cfg['video']['window_frames'], augment=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for frames, labels in loader:
            out = model(frames.to(device))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    uar = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f'  acc={acc:.4f}  macro-F1={f1:.4f}  UAR={uar:.4f}')

    cm = confusion_matrix(all_labels, all_preds)
    names = [IDX_TO_LABEL[i] for i in range(cfg['emotions']['num_classes'])]
    print('  Confusion matrix:')
    print(pd.DataFrame(cm, index=names, columns=names).to_string())
    print()
