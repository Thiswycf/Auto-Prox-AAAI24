# 自动化代理公式汇总（work_dirs/autoprox_search）

## 6 个任务与对应 repr_geno

| 搜索空间 | 数据集 | repr_geno |
|----------|--------|-----------|
| AutoFormerSub | Chaoyang | `INPUT:(t2,dss,er)TREE:(abs)BINARY:(element_wise_sum)` |
| AutoFormerSub | CIFAR-100 | `INPUT:(er,nwot)TREE:(element_wise_square)BINARY:(element_wise_sub)` |
| AutoFormerSub | Flowers | `INPUT:(er,t2,nwot)TREE:(frobenius_norm)BINARY:(element_wise_mul)` |
| PiT | Chaoyang | `INPUT:(er,t2)TREE:(element_wise_sqrt\|frobenius_norm\|element_wise_square)BINARY:(element_wise_sum)` |
| PiT | CIFAR-100 | `INPUT:(t2,nwot,er)TREE:(log\|abs\|frobenius_norm)BINARY:(element_wise_sum)` |
| PiT | Flowers | `INPUT:(er,t2,fisher)TREE:(element_wise_square)BINARY:(element_wise_mul)` |

符号说明：$\mathcal{E}$=ER, $\mathcal{T}_2$=t2, $\mathcal{D}$=DSS, $\mathcal{N}$=NWOT, $\mathcal{F}$=Fisher；$\|\cdot\|_F$=Frobenius 范数；$\odot$=逐元素乘。

---

## LaTeX 表格（2 行 × 3 列）

```latex
\begin{table*}[t]
\centering
\caption{Automated ZCP formulas obtained from searches in various search spaces and datasets}
\label{tab:autoprox_zcp}
\begin{tabular}{lcc}
\toprule
\textbf{Dataset} & \textbf{AutoFormerSub} & \textbf{PiT} \\
\midrule
Chaoyang
& $\displaystyle\sum_l \left( |\mathrm{T2}^{(l)}| + |\mathrm{DSS}^{(l)}| + |\mathrm{ER}^{(l)}| \right)$
& $\displaystyle\sum_l \left( \|\sqrt{\mathrm{ER}^{(l)}}\|_F^{2} + \|\sqrt{\mathrm{T2}^{(l)}}\|_F^{2} \right)$ \\
\addlinespace
CIFAR-100
& $\displaystyle\sum_l \left( (\mathrm{ER}^{(l)})^2 - (\mathrm{NWOT}^{(l)})^2 \right)$
& $\displaystyle\sum_l \left( \|\left|\log(\mathrm{T2}^{(l)})\right|\|_F + \|\left|\log(\mathrm{NWOT}^{(l)})\right|\|_F + \|\left|\log(\mathrm{ER}^{(l)})\right|\|_F \right)$ \\
\addlinespace
Flowers
& $\displaystyle\sum_l \left( \|\mathrm{ER}^{(l)}\|_F \odot \|\mathrm{T2}^{(l)}\|_F \odot \|\mathrm{NWOT}^{(l)}\|_F \right)$
& $\displaystyle\sum_l \left( (\mathrm{ER}^{(l)})^2 \odot (\mathrm{T2}^{(l)})^2 \odot (\mathrm{Fisher}^{(l)})^2 \right)$ \\
\bottomrule
\end{tabular}
\end{table*}
```

若表格中希望每格只保留最简公式（不写“逐层”和 $\phi/\psi$ 的说明），可使用下面紧凑版：

```latex
\begin{table}[t]
\centering
\caption{各搜索空间与数据集上搜索得到的自动化 ZCP 公式（紧凑版）}
\begin{tabular}{lccc}
\toprule
\textbf{搜索空间} & \textbf{Chaoyang} & \textbf{CIFAR-100} & \textbf{Flowers} \\
\midrule
AutoFormerSub
& $\sum \left( |\mathcal{T}_2| + |\mathcal{D}| + |\mathcal{E}| \right)$
& $\mathcal{E}^2 - \mathcal{N}^2$
& $\|\mathcal{E}\|_F \odot \|\mathcal{T}_2\|_F \odot \|\mathcal{N}\|_F$ \\
\addlinespace
PiT
& $\phi(\mathcal{E}) + \phi(\mathcal{T}_2)$, $\phi \in \{\sqrt{\cdot}, \|\cdot\|_F, (\cdot)^2\}$
& $\psi(\mathcal{T}_2) + \psi(\mathcal{N}) + \psi(\mathcal{E})$, $\psi \in \{\log, |\cdot|, \|\cdot\|_F\}$
& $\mathcal{E}^2 \odot \mathcal{T}_2^2 \odot \mathcal{F}^2$ \\
\bottomrule
\end{tabular}
\end{table}
```

需在导言区使用：`\usepackage{booktabs}`（若未使用可去掉 `\toprule/\midrule/\bottomrule` 改为 `\hline`）。
