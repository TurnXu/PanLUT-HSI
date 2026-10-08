# PanLUT-HSI

本工程把 Pan-LUT 迁移到“16波段高光谱马赛克 + 单通道高分辨率PAN”任务，并沿用现有数据工程的训练、生成、测试和可视化命令。

核心数据流：

```text
单通道4×4 MSFA马赛克
  → PixelUnshuffle(4)，得到16通道低分辨率光谱
  → 无bias的1×1卷积：16→4
  → 上采样并拼接PAN，形成5通道输入
  → PGLUT → 四方向SDLUT → AOLUT
  → 无bias的1×1卷积：4→16
  → 高分辨率16波段融合结果
```

单个光谱波段相对PAN的有效空间倍率为8；单通道马赛克相对PAN的倍率为2。

## 文件排列

工程入口按照现有数据处理模板平铺，便于逐文件对照：

```text
PanLUT-HSI/
├── model.py
├── lut_layers.py
├── official_ops.py
├── GetDataSet.py
├── degradation.py
├── losses.py
├── utils.py
├── quality_index.py
├── q2n.py
├── train_simulate.py
├── train_realworld.py
├── generate.py
├── test.py
├── visualize.py
├── test_for_degradation.py
├── degrade_visualize.py
├── generate_for_video.py
├── get_param_and_flops.py
└── flops_counter.py
```

`configs/`保存JSON配置，`tests/`保存自动化测试；算法实现不再放在额外的Python包目录中。

## 数据目录

从本工程目录运行命令时，默认数据目录为 `../Dataset`：

```text
Dataset/
├── CAVE/
│   ├── train/
│   └── test/
├── chikusei/
│   ├── train/
│   └── test/
└── real_world/
    ├── train/
    │   ├── mosaic/
    │   └── pan/
    └── test/
        ├── mosaic/
        └── pan/
```

CAVE读取MAT变量 `b` 的第12至27号波段；Chikusei读取变量 `chikusei`。真实RAW按照12 bit、`255×276`基础光谱平面布局解析。

## 1. 训练

当前电脑只有一张GPU，示例统一使用 `--device 0`。服务器训练时可替换成实际GPU编号。

### CAVE

```powershell
python train_simulate.py --idx 1 --data_path ../Dataset --dataset CAVE --epochs 1000 --train_size 64 --stride 32 --batch_size 16 --lr_decay --save_freq 50 --device 0
```

### Chikusei

```powershell
python train_simulate.py --idx 2 --data_path ../Dataset --dataset chikusei --epochs 40 --train_size 64 --stride 64 --batch_size 16 --lr_decay --save_freq 5 --device 0
```

### Real-world

```powershell
python train_realworld.py --idx 3 --data_path ../Dataset --dataset real_world --epochs 10 --train_size 64 --stride 64 --batch_size 16 --lr_decay --save_freq 1 --visual --visual_freq 1 --device 0
```

训练模型保存到：

```text
<idx>/model/best.pth
<idx>/model/best_<epoch>.pth
<idx>/model/<epoch>.pth
```

仍可使用JSON配置入口：

```powershell
python train_simulate.py --config configs/baseline.json --device 0
python train_realworld.py --config configs/realworld.json --device 0
```

## 2. Generate

结果保存到：

```text
<dataset>/<idx>/result/mat/
├── fused/
├── mosaic/
├── pan/
└── gt/       # 仅模拟数据
```

其中 `mosaic` 是从单通道MSFA排列解包得到的16通道低分辨率数据，与原工程保持一致。

### CAVE：生成全部测试数据

```powershell
python generate.py --idx 1 --simulate --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../Dataset --dataset CAVE --load_model ./1/model/best.pth --device 0
```

### CAVE：生成单个样本

```powershell
python generate.py --idx 1 --simulate --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../Dataset --dataset CAVE --load_model ./1/model/best.pth --data_id watercolors_ms.mat --device 0
```

### Chikusei：生成全部测试数据

```powershell
python generate.py --idx 2 --simulate --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../Dataset --dataset chikusei --load_model ./2/model/best.pth --device 0
```

### Chikusei：生成单个样本

```powershell
python generate.py --idx 2 --simulate --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../Dataset --dataset chikusei --load_model ./2/model/best.pth --data_id 2.mat --device 0
```

### Real-world：生成全部测试数据

```powershell
python generate.py --idx 3 --real_world --mosaic_save --pan_save --demosaic_save --data_path ../Dataset --dataset real_world --load_model ./3/model/best.pth --device 0
```

### Real-world：生成指定样本

```powershell
python generate.py --idx 3 --real_world --mosaic_save --pan_save --demosaic_save --data_path ../Dataset --dataset real_world --load_model ./3/model/best.pth --data_id 1.raw 16.raw 37.raw --device 0
```

## 3. 测试

模拟数据计算 PSNR、SSIM、SAM 和 ERGAS；真实数据计算 QNR、Dλ 和 Ds。

### CAVE

```powershell
python test.py --idx 1 --simulate --data_path ./CAVE/1/result/mat/
python test.py --idx 1 --simulate --data_path ./CAVE/1/result/mat/ --data_id oil_painting_ms.mat
```

### Chikusei

```powershell
python test.py --idx 2 --simulate --data_path ./chikusei/2/result/mat/
python test.py --idx 2 --simulate --data_path ./chikusei/2/result/mat/ --data_id 2.mat
```

### Real-world

```powershell
python test.py --idx 3 --real_world --data_path ./real_world/3/result/mat/ --patch_size 64
python test.py --idx 3 --real_world --data_path ./real_world/3/result/mat/ --patch_size 64 --data_id 1.mat
```

## 4. 可视化

### CAVE融合结果

```powershell
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/fused --save_path ./CAVE/1/result/rgb/fused --data_type fused --data_id watercolors_ms.mat --detach --detach_size 50 50 --detach_coordinate 160 210 160 210 --boxcolor b --boxwidth 2
```

### CAVE误差图

```powershell
python visualize.py --visual_task diffmap --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/fused --mat_path_for_diff ./CAVE/1/result/mat/gt --save_path ./CAVE/1/result/rgb/diffmap --data_id watercolors_ms.mat --detach --detach_size 50 50 --detach_coordinate 160 210 160 210 --boxcolor b --boxwidth 2 --mae_level 64 --sam_level 64
```

### CAVE上采样马赛克、PAN和GT

```powershell
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/mosaic --save_path ./CAVE/1/result/rgb/upmosaic --data_type upmosaic --data_id watercolors_ms.mat
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/pan --save_path ./CAVE/1/result/rgb/pan --data_type pan --data_id watercolors_ms.mat
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/gt --save_path ./CAVE/1/result/rgb/gt --data_type gt --data_id watercolors_ms.mat
```

Chikusei和真实数据只需替换数据集名称、实验编号、样本名及局部区域坐标。例如真实数据：

```powershell
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/3/result/mat/fused --save_path ./real_world/3/result/rgb/fused --data_id 1.mat --data_type fused --detach --detach_size 128 128 --detach_coordinate 700 900 700 900 --boxwidth 5 --boxcolor b
```

## LUT后端与注意事项

- `model.backend=torch` 是纯PyTorch多线性插值实现，适合验证正确性，但整幅真实图推理较慢。
- `model.backend=official_cuda` 使用 `Pan-LUT-main` 中已经编译好的官方CUDA扩展；本工程不会自动编译或修改官方算子。
- 两个新增的1×1卷积均可学习且 `bias=False`。
- 五维LUT采用分块像素计算降低峰值显存，但 README 中的批大小仍需按实际显存调整。

## 5. 辅助工具

### 验证模拟观测退化是否一致

不加载模型时，直接检查GT生成的马赛克与PAN：

```powershell
python test_for_degradation.py --simulate --dataset CAVE --data_path ../Dataset --data_id watercolors_ms.mat --device 0
```

加载模型后检查融合结果的观测重建误差：

```powershell
python test_for_degradation.py --real_world --dataset real_world --data_path ../Dataset --load_model ./3/model/best.pth --data_id 1.raw --device 0
```

### 可视化退化结果

```powershell
python degrade_visualize.py --data_path ./CAVE/1/result/mat --dataset CAVE --save_path ./CAVE/1/result/rgb/degradation
python degrade_visualize.py --data_path ./real_world/3/result/mat --dataset real_world --load_model ./3/model/best.pth --save_path ./real_world/3/result/rgb/degradation
```

### 多帧RAW生成

`generate_for_video.py`用于一个RAW文件中连续保存多帧的情况；普通单帧数据仍使用 `generate.py`。

```powershell
python generate_for_video.py --real_world_video --idx 3 --dataset real_world --data_path ../Dataset --load_model ./3/model/best.pth --frames 0 1 2 --mosaic_save --pan_save --device 0
```

脚本会检查RAW总长度、mosaic/PAN帧数和 `--frames` 边界，输出名称形如 `1_0.mat`。

### 参数量与理论运算规模

```powershell
python get_param_and_flops.py --config configs/baseline.json --batch_size 1 --height 512 --width 512
```

工具分别报告1×1卷积MACs与LUT顶点访问次数，不把查表访问错误地等价成卷积FLOPs。

## 最小验证

```powershell
python -X utf8 -m compileall .
python -X utf8 -m unittest discover -s tests -v
```
