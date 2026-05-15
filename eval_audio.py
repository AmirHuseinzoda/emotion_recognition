import sys; sys.path.insert(0,'.')
import torch, yaml, pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
from pathlib import Path
from datasets.ravdess import RAVDESSAudioDataset, IDX_TO_LABEL_6 as IDX_TO_LABEL
from models.audio.transformer import AudioEmotionModel

cfg = yaml.safe_load(open('configs/config.yaml'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

model = AudioEmotionModel(
    num_classes=cfg['emotions']['num_classes'],
    model_name=cfg['audio']['model_name'],
    dropout=0.0,
)
model.load_state_dict(torch.load('checkpoints/audio_best.pt', map_location='cpu', weights_only=False))
model.eval().to(device)

audio_cfg = cfg['audio']

for split in ['val', 'test']:
    df = pd.read_csv(Path(cfg['data']['splits_dir']) / f'{split}.csv')
    df = df[(df['ext'] == '.wav') & (df['source'] == 'cremad')].reset_index(drop=True)
    print(f'{split} samples: {len(df)}')

    ds = RAVDESSAudioDataset(
        df,
        target_sr=audio_cfg['sample_rate'],
        max_duration_sec=audio_cfg['max_duration_sec'],
        augment=False,
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for waveform, labels in loader:
            out = model(waveform.to(device))
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
