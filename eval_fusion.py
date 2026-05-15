import sys; sys.path.insert(0, '.')
import torch, yaml, pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
from pathlib import Path
from datasets.ravdess import (RAVDESSAudioDataset, RAVDESSVideoDataset,
                               RAVDESSMultimodalDataset, IDX_TO_LABEL_6 as IDX_TO_LABEL)
from models.video.backbone import VideoEmotionModel
from models.audio.transformer import AudioEmotionModel
from models.fusion.fusion import FusionModel

cfg = yaml.safe_load(open('configs/config.yaml'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tcn = cfg['video']['tcn']
num_classes = cfg['emotions']['num_classes']

video_model = VideoEmotionModel(num_classes=num_classes, num_channels=tcn['num_channels'],
    num_levels=tcn['num_levels'], kernel_size=tcn['kernel_size'], dropout=0.0, pretrained=False)
video_model.load_state_dict(torch.load('checkpoints/video_best.pt', map_location='cpu', weights_only=False))

audio_model = AudioEmotionModel(num_classes=num_classes, model_name=cfg['audio']['model_name'], dropout=0.0)
audio_model.load_state_dict(torch.load('checkpoints/audio_best.pt', map_location='cpu', weights_only=False))

fusion_model = FusionModel(video_model=video_model, audio_model=audio_model,
    fusion_type='attention', num_classes=num_classes,
    video_embed_dim=tcn['num_channels'], audio_embed_dim=cfg['audio']['hidden_size'],
    hidden_dim=cfg['fusion']['hidden_dim'])
fusion_model.load_state_dict(torch.load('checkpoints/fusion_best.pt', map_location='cpu', weights_only=False))
fusion_model.eval().to(device)
print('Models loaded OK')

proc_dir = Path(cfg['data']['processed_dir']) / 'video'
audio_cfg = cfg['audio']

def _pair_cremad(df_all, proc_video_dir):
    if 'source' not in df_all.columns: return pd.DataFrame(), pd.DataFrame()
    c_vid = df_all[(df_all['ext'] == '.flv') & (df_all['source'] == 'cremad')].copy()
    c_aud = df_all[(df_all['ext'] == '.wav') & (df_all['source'] == 'cremad')].copy()
    if c_vid.empty or c_aud.empty: return pd.DataFrame(), pd.DataFrame()
    v_stems = c_vid['path'].apply(lambda p: Path(p).stem)
    a_stems = c_aud['path'].apply(lambda p: Path(p).stem)
    common = sorted(set(v_stems) & set(a_stems))
    c_vid = (c_vid[v_stems.isin(common)].assign(_s=lambda d: d['path'].apply(lambda p: Path(p).stem))
             .sort_values('_s').drop(columns='_s').reset_index(drop=True))
    c_aud = (c_aud[a_stems.isin(common)].assign(_s=lambda d: d['path'].apply(lambda p: Path(p).stem))
             .sort_values('_s').drop(columns='_s').reset_index(drop=True))
    npy_mask = c_vid['path'].apply(lambda p: (proc_video_dir / (Path(p).stem + '.npy')).exists())
    missing = set(c_vid[~npy_mask]['path'].apply(lambda p: Path(p).stem))
    c_vid = c_vid[npy_mask].reset_index(drop=True)
    c_aud = c_aud[~c_aud['path'].apply(lambda p: Path(p).stem).isin(missing)].reset_index(drop=True)
    return c_vid, c_aud

for split in ['val', 'test']:
    df_all = pd.read_csv(Path(cfg['data']['splits_dir']) / f'{split}.csv')
    df_vid, df_aud = _pair_cremad(df_all, proc_dir)
    print(f'\n{split}: {len(df_vid)} pairs')

    vid_ds = RAVDESSVideoDataset(df_vid, proc_dir, window_frames=cfg['video']['window_frames'], augment=False)
    aud_ds = RAVDESSAudioDataset(df_aud, target_sr=audio_cfg['sample_rate'],
                                  max_duration_sec=audio_cfg['max_duration_sec'], augment=False)
    mm_ds = RAVDESSMultimodalDataset(aud_ds, vid_ds)
    loader = DataLoader(mm_ds, batch_size=16, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for frames, waveform, labels in loader:
            out = fusion_model(frames.to(device), waveform.to(device))
            all_preds.extend(out['logits_fusion'].argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    uar = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f'  Fusion  acc={acc:.4f}  macro-F1={f1:.4f}  UAR={uar:.4f}')
    cm = confusion_matrix(all_labels, all_preds)
    names = [IDX_TO_LABEL[i] for i in range(num_classes)]
    print(pd.DataFrame(cm, index=names, columns=names).to_string())
