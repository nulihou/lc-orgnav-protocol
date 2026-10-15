# P3 多种子 + 等价检验 — 冻结协议（2026-10-13，任何新训练与结果之前）
- 用户未答复 δ；按既定流程在看结果前选定：**主等价界 δ = 2.0 个百分点（绝对 ProtoAcc@1）**，**敏感性 δ = 1.0 pp**。理由见 PLAN_P3_DRAFT.md（3-seed 现有差距 0.7–1.6 pp；300 组下单方法 SE 约 0.5–1.1 pp；2 pp 为作者判断的实际可忽略上限）。文中须写明 δ 为事先选定。
- 新训练种子：1, 3, 5, 11, 13, 17, 19（加现有 7/21/42 共 10）。不新增其它种子，不丢弃任何种子。
- 训练：E:\conda_envs\lcorgnav_cuda_full\python.exe -m r3_selectors.pipeline --root <本目录>\root --track {scale300,hard_confusable} --seeds 1 3 5 11 13 17 19，其余参数为默认（epochs 80, batch 64, lr 3e-4, 原容量）；root\02_protocol 为指向原 02_protocol 的目录联接（只读使用），输出只写本目录 root\04_results。原 04_results 不动。
- 模型选择：沿用 pipeline 规则（验证准确率→验证 CE），不看测试集。
- 分析（独立计分脚本，在全部 test_predictions 哈希登记后运行）：每方法 10-seed ProtoAcc 均值/SD；Graph 与每个对照的配对差（先在每个 group 内对 10 seed 求平均正确率，再求组均值差）；group cluster bootstrap 10k（seed 20261002）给 95% CI 和 90% CI；TOST：90% CI ⊂ (−δ, +δ) 即判等价（δ=2 主、δ=1 敏感性）；同时报告差值 95% CI 与双侧 sign test + Holm（每轨 3 项）。
- 结果无论支持与否均全部报告；TOST 不成立时只能写"未能确认等价"。
