# 008-3验收（eval n-20261004-115031-486 ← impl n-20261004-105422-1c7）

口径依据：008续-3 spec 四条（裸对裸P1对照 + MedSAM/SAM短预算 + 作废声明 + 权重进容器）+ eval_008b 转正三判据。

## 1. 结论：P1转正（裸HD95主判据通过；test Dice反超作废；C/D链路阻塞）

### 1.1 裸对裸P1对照（四ckpt test50裸mask，关后处理，本地独立重跑）

| ckpt | val | test 裸Dice | test 裸HD95 cdist/mirror |
|---|---|---|---|
| UNeXt grid1-s0（epoch28/sha f5e41cc8） | 0.9120 | 0.9110 | 11.00 / 12.76 |
| UNeXt grid1-s1（epoch28/sha f5e41cc8） | 0.9100 | 0.9116 | 10.54 / 11.62 |
| DINO s0（epoch28/sha 1053c8f1） | 0.9188 | 0.9082 | 7.32 / 7.70 |
| DINO s1（epoch29/sha 1053c8f1） | 0.9216 | 0.9085 | 7.65 / 8.14 |

- verdict看裸HD95：DINO好 ~3.5（cdist均值7.5 vs 10.8；mirror均值7.9 vs 12.2）✅——与parent预期一致。
- test Dice反超作废：裸口径下DINO Dice仍低 ~0.3pt（0.908 vs 0.911）；后处理档UNeXt +0.27pt（0.9110→0.9136，grid1_lock）而DINO零增益（008b读数），故"UNeXt后处理后反超"是管线artifact——作废声明成立，转正判据剔除test Dice反超项。
- seed一致性：DINO双seed test差0.0003（饱和一致）；UNeXt双seed差0.0006——方向同，幅度在带内。
- split_md5=682f3d89四run一致 ✅；ckpt keys含ds_head+frame_only ✅；anchor参数量UNeXt 7.6M / DINO 22M ✅。

### 1.2 全预算FO对照（沿用已验收值，不重跑）

DINO-FO 0.8928 vs UNeXt-FO 0.8025（同400/30ep，val +9.0pt）——eval_008b已验收，本次不重跑，直接引用。

### 1.3 P1转正判定

| 判据（作废后） | 结果 |
|---|---|
| 裸HD95（主） | DINO好~3.5，双seed同向 ✅ |
| 全预算FO对照 | +9.0pt ✅ |
| seed一致性 | 双seed同向，test差<0.001 ✅ |

P1转正 ✅（test Dice反超项已作废，不再列判据）。

## 2. C/D（MedSAM/SAM）：链路阻塞，判医学有用不可行

- 接口：`src/diag/sam_anchor.py`（commit b4fcc24）读验一致——frozen + vision_encoder直出(1,256,64,64) + 1×1 adapt 256→96 + bilinear×2至F_tf；train接线`--anchor medsam|sam` ✅。
- 权重：本地blobs在（models--wanglab--medsam-vit-base / models--facebook--sam-vit-base），但均为40–72K指针文件，**真权重未下载**（1.5G blobs系008b旧记录，现行cache无大文件）。
- 远端：:31035 sam_fo双log均为`torch.OutOfMemoryError`（1024输入×4，decomposed_rel_pos reshape时7.5GiB分配失败，19.5GiB已占）——medsam FAILED（val 0.7458/40.8，4pts）/ sam FAILED（0.7582/41.9，4pts）；另各一RUNNING僵尸行（旧轮残留，以FAILED为准）。
- 判定：输入1024×4路径在A30 24G上不可行（batch2 + PCLF/HDC/Decoder同卡），且超声256→1024上采样×4本身是模糊+16x像素偏差——C/D记**链路阻塞**（权重未进容器 + 显存不可行），判"医学有用"不可行，不追全预算。 mediana通用SAM对照同样阻塞。
- 后继方向（二选一）：① 降输入（512/384）重探显存；② 放弃SAM线，P1链以DINO+Medsam文档级结论关闭。

## 3. 与论文/H结论的对应关系

- H1分母重标：DINO-FO 0.8928取代UNeXt-FO 0.8025；全链−FO同锚内差待后继（DINO全链0.908 vs FO 0.888；UNeXt全链0.911 vs FO 0.808）。
- P1链条更新：val/test方向分裂（+0.7/−0.5）已解释为管线artifact——裸口径下DINO Dice仍低，HD95才是真差。作废声明追溯至本文件§1.1。
- 不产生Table对照数值（裸口径是诊断口径，报方向不报点）。
