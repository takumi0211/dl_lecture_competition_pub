import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
import random
import numpy as np
from src.models.evflownet import EVFlowNet
from src.datasets import DatasetProvider
from enum import Enum, auto
from src.datasets import train_collate
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
import os
import time
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def compute_epe_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor):
    '''
    end-point-error (ground truthと予測値の二乗誤差)を計算
    pred_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 予測したオプティカルフローデータ
    gt_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 正解のオプティカルフローデータ
    '''
    epe = torch.mean(torch.mean(torch.norm(pred_flow - gt_flow, p=2, dim=1), dim=(1, 2)), dim=0)
    return epe

# loss関数の変更
def compute_multiscale_loss(pred_flows: Dict[str, torch.Tensor], gt_flow: torch.Tensor) -> torch.Tensor:
    total_loss = 0
    scales = {'flow0': 0.125, 'flow1': 0.25, 'flow2': 0.5, 'flow3': 1}
    weights = scales

    for scale, factor in scales.items():
        # 正解データを予測データのサイズにリサイズ
        scaled_gt = F.interpolate(gt_flow, size=pred_flows[scale].shape[2:], mode='bicubic', align_corners=True) # modeをbase.pyに揃える
        loss = compute_epe_error(pred_flows[scale], scaled_gt)
        total_loss += loss * weights[scale]

    return total_loss

def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    '''
    optical flowをnpyファイルに保存
    flow: torch.Tensor, Shape: torch.Size([2, 480, 640]) => オプティカルフローデータ
    file_name: str => ファイル名
    '''
    np.save(f"{file_name}.npy", flow.cpu().numpy())

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    '''
        ディレクトリ構造:

        data
        ├─test
        |  ├─test_city
        |  |    ├─events_left
        |  |    |   ├─events.h5
        |  |    |   └─rectify_map.h5
        |  |    └─forward_timestamps.txt
        └─train
            ├─zurich_city_11_a
            |    ├─events_left
            |    |       ├─ events.h5
            |    |       └─ rectify_map.h5
            |    ├─ flow_forward
            |    |       ├─ 000134.png
            |    |       |.....
            |    └─ forward_timestamps.txt
            ├─zurich_city_11_b
            └─zurich_city_11_c
        '''
    
    # ------------------
    #    Dataloader
    # ------------------
    loader = DatasetProvider(
        dataset_path=Path(args.dataset_path),
        representation_type=RepresentationType.VOXEL,
        delta_t_ms=100,
        num_bins=4
    )
    train_set = loader.get_train_dataset()
    test_set = loader.get_test_dataset()
    collate_fn = train_collate

    # トレーニングデータをトレーニングと検証に分割する
    train_indices, val_indices = train_test_split(
        list(range(len(train_set))),
        test_size=0.2,
        random_state=args.seed
    )

    train_subset = Subset(train_set, train_indices)
    val_subset = Subset(train_set, val_indices)

    train_data = DataLoader(train_subset,
                            batch_size=args.data_loader.train.batch_size,
                            shuffle=args.data_loader.train.shuffle,
                            collate_fn=collate_fn,
                            drop_last=False)

    val_data = DataLoader(val_subset,
                          batch_size=args.data_loader.test.batch_size,
                          shuffle=False,  # 検証データはシャッフルしない
                          collate_fn=collate_fn,
                          drop_last=False)

    test_data = DataLoader(test_set,
                           batch_size=args.data_loader.test.batch_size,
                           shuffle=args.data_loader.test.shuffle,
                           collate_fn=collate_fn,
                           drop_last=False)

    '''
    train data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
        Key: flow_gt, Type: torch.Tensor, Shape: torch.Size([Batch, 2, 480, 640]) => オプティカルフローデータのバッチ
        Key: flow_gt_valid_mask, Type: torch.Tensor, Shape: torch.Size([Batch, 1, 480, 640]) => オプティカルフローデータのvalid. ベースラインでは使わない
    
    test data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
    '''
    # ------------------
    #       Model
    # ------------------
    model = EVFlowNet(args.train).to(device)

    # ------------------
    #   optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.initial_learning_rate, weight_decay=args.train.weight_decay)
    
    # Create the directory if it doesn't exist
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    min_val_loss = float('inf')  # 最小の検証ロスを追跡するための変数
    
    # ------------------
    #   Start training
    # ------------------
    for epoch in range(args.train.epochs):
        total_loss = 0
        val_loss = 0
        print("on epoch: {}".format(epoch+1))

        # トレーニングループ
        model.train()
        for i, batch in enumerate(tqdm(train_data)):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device)  # [B, 4, 480, 640]
            ground_truth_flow = batch["flow_gt"].to(device)  # [B, 2, 480, 640]
            flow_dict, final_flow = model(event_image)  # [B, 2, 480, 640]
            loss: torch.Tensor = compute_multiscale_loss(flow_dict, ground_truth_flow)
            final_loss: torch.Tensor = compute_epe_error(final_flow, ground_truth_flow)
            print(f"batch {i} loss: {loss.item()}, final loss: {final_loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_data)}')

        # 検証ループ
        model.eval()
        val_loss = 0  # 検証ロスの初期化
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_data)):
                batch: Dict[str, Any]
                event_image = batch["event_volume"].to(device)  # [B, 4, 480, 640]
                ground_truth_flow = batch["flow_gt"].to(device)  # [B, 2, 480, 640]
                flow_dict, final_flow = model(event_image)  # [B, 2, 480, 640]
                loss: torch.Tensor = compute_multiscale_loss(flow_dict, ground_truth_flow)
                final_loss: torch.Tensor = compute_epe_error(final_flow, ground_truth_flow)
                val_loss += loss.item()
            val_loss /= len(val_data)
            print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')

            # 最小の検証ロスを持つモデルを保存
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                current_time = time.strftime("%Y%m%d%H%M%S")
                model_path = f"checkpoints/model_{current_time}_best.pth"
                torch.save(model.state_dict(), model_path)
                print(f"epoch:{epoch+1}, Best model saved to {model_path}")

    # 最後にベストモデルをロードしてテストデータに対して予測を行う
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    flow: torch.Tensor = torch.tensor([]).to(device)
    with torch.no_grad():
        print("start test")
        for batch in tqdm(test_data):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device)
            _ , batch_flow = model(event_image) # [1, 2, 480, 640]
            flow = torch.cat((flow, batch_flow), dim=0)  # [N, 2, 480, 640]
        print("test done")
    # ------------------
    #  save submission
    # ------------------
    file_name = "submission"
    save_optical_flow_to_npy(flow, file_name)

if __name__ == "__main__":
    main()
