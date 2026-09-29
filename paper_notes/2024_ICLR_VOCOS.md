# Vocos 论文阅读笔记：从 mel 频谱到复数 STFT，再到波形

> **论文**：Hubert Siuzdak, *Vocos: Closing the Gap Between Time-Domain and Fourier-Based Neural Vocoders for High-Quality Audio Synthesis*, ICLR 2024  
> **论文链接**：[OpenReview](https://openreview.net/forum?id=d6b6fd2b9464f306e29d42b554de0a493bb52ade) · [arXiv](https://arxiv.org/abs/2306.00814)  
> **关键词**：神经声码器、复数 STFT、相位缠绕、iSTFT、GAN  
> **代码**：[gemelo-ai/vocos](https://github.com/gemelo-ai/vocos)

## 一、总起：这篇文章做了什么？

Vocos 是一个把 **mel 频谱等声学特征转换为音频波形**的神经声码器。常见的时域 GAN 声码器利用多层转置卷积，逐级把低帧率特征上采样成高采样率波形。Vocos 选择另一条路：网络保持输入的帧级时间分辨率，直接预测每一帧的**完整复数 STFT 系数**，再用固定的逆短时傅里叶变换（iSTFT）合成波形。因此，它一次前向传播即可出声，不涉及扩散或 flow matching 的迭代采样。

模型生成复数谱时分开预测幅度参数 $m$ 和相位参数 $p$，令


```math

\hat S=\exp(m)\,[\cos(p)+j\sin(p)].

```


这个参数化尊重相位的周期性，不强制网络在 $(-\pi,\pi]$ 内回归一个会在边界突跳的角度。Vocos 通过生成波形的 mel 重建损失、对抗损失和特征匹配损失训练；**没有逐点的真实相位监督损失**。它学习的是能产生高质量波形的相位，而非保证逐点还原原录音的唯一相位。


默认 mel 路径可以按下面的形状追踪。设 mel 有 $T$ 帧，配置使用 24 kHz、100 个 mel bin、`n_fft=1024`、`hop_length=256`：
（Vocos 默认采样率是 24,000 Hz，所以 n_fft=1024 个采样点约为 42.7 ms）
```text
mel [B, 100, T]
  → ConvNeXt 主干 [B, T, 512]（保持 T 帧）
  → Linear(512, 1026) [B, T, 1026]
  → 每帧 513 个 m + 513 个 p
  → 复数 STFT [B, 513, T]
  → iSTFT → 波形 [B, 约 T×256]
```

`513=1024/2+1`，来自实数信号傅里叶谱的共轭对称性。网络并非自己发现“1026 个数是声音”：输出形状与 iSTFT 是研究者按信号处理规则设定的；训练再学习如何从 mel 预测这些系数。

## 二、阅读中的 Q&A


```diff
- Q1：文章提到的 phase 难点是什么？
```

**A：** STFT 的一个时频点是复数 $S=M e^{j\phi}$：幅度 $M$ 表示该频率有多强，相位 $\phi$ 表示相应波动的位置。相位是周期量，通常显示为 $(-\pi,\pi]$。真实相位从 $179^\circ$ 前进到 $181^\circ$ 只走了 $2^\circ$，主值却记录成 $179^\circ\to-179^\circ$。波形并没有跳，**角度的记法跳了**。若把这些主值当普通实数做 MSE，两个几乎相同的方向会被当成相差 $358^\circ$。相位还需与相邻时频点协同，才能生成自然波形。

```diff
- Q2：Vocos 怎样处理 phase wrapping？
```

**A：** Head 输出实数 $p$，通过 $\cos p,\sin p$ 构成单位圆上的方向，再与正的幅度相乘：


```math

M=\exp(m),\qquad \hat S=M(\cos p+j\sin p).

```


网络不直接回归限定在 $(-\pi,\pi]$ 的相位主值，也不对主值逐点做 MSE。`cos/sin` 让 $p$ 与 $p+2\pi$ 得到相同复数方向。随后 iSTFT 得到波形，损失从最终声音反向传播。论文把这种处理称为**隐式相位缠绕**；它缓解边界造成的优化问题，并不宣称唯一恢复原始相位。

```diff
- Q3：p 仍然是角度，为什么不会遇到同样的边界？
```

**A：** 要区分**网络内部的 $p$** 与**显示出来的相位主值**。若连续转动跨过 $180^\circ$，网络可以输出 $179^\circ\to181^\circ$；只有额外调用 `atan2(sin(p), cos(p))`、要求显示在 $(-180^\circ,180^\circ]$ 内时，才会出现 $179^\circ\to-179^\circ$ 的记数跳变。生成 STFT 的代码直接使用 `cos(p)`、`sin(p)`，不需要把 `atan2` 的主值再交回网络。

```diff
- Q4：如果不限制，p 不是会无限大吗？网络输出不需要边界才能收敛吗？代码实际怎么写？
```

**A：** `ISTFTHead` 的输出是普通线性层，将每帧隐藏特征投影成幅度参数和相位参数；对 **`p` 没有 `tanh`、`clip` 或固定角度区间**。官方实现的关键逻辑如下（变量名略作整理）：

```python
head_output = self.out(hidden).transpose(1, 2)
mag_logits, p = head_output.chunk(2, dim=1)
magnitude = torch.exp(mag_logits).clip(max=1e2)
complex_stft = magnitude * (torch.cos(p) + 1j * torch.sin(p))
waveform = self.istft(complex_stft)
```

这里的 `clip(max=1e2)` **只限制幅度**，不限制 $p$。收敛也不要求输出有硬边界：神经网络的线性层可以输出实数，却通常会在训练中找到有限值。对周期方向，$p$、$p+2\pi$ 是等价解；优化器只需找到其中一个有用的有限解，不会因此必须无限增长。实际损失施加在合成波形上，而非要求 $p$ 收敛到某个指定角度。极端大的 $p$ 可能带来浮点精度问题，但论文和默认代码没有为 $p$ 设置固定裁剪范围。

```diff
- Q5：判别器MPD 是按不同 phase 逆向生成声音吗？MRD 只输入一个点吗？输入尺寸变化怎么处理？
```

**A：** **两者都只是训练时的判别器，不生成声音。** 生成器 Head 先预测复数 STFT 并经 iSTFT 得到 $\hat x$；真实波形 $x$ 与生成波形 $\hat x$ 才分别进入判别器。

| 判别器 | 实际输入处理 | 不是在做什么 |
|---|---|---|
| **MPD** | 将**整段波形**按 `period=(2,3,5,7,11)` 分别重排为二维阵列；每种重排有一个卷积子判别器 | 这里的 *period* 是重排的采样点数，**不是 phase**，也不是按五种相位生成五段音频 |
| **MRD** | 对**整段波形**分别做 `n_fft=(512,1024,2048)` 的滑窗 STFT；每种分辨率得到一整张时间 × 频率的复数图，再将实部和虚部作为 2 个通道输入对应卷积子判别器 | “1024 点”是每个分析窗的长度，**不是只给网络一个采样点** |

例如 12 个波形采样点按 `period=3` 重排为 4 行 × 3 列；MPD 的卷积在重排后的局部结构上判断真假。MRD 用 1024 点窗时，一个时间窗有 513 个单侧频率格，窗口沿**整段**声音滑动会得到许多时间帧。

不同 `n_fft` 的图尺寸本来就不同，但 **512、1024、2048 各由独立子判别器处理**。它们使用卷积，不要求固定的时间帧数；输出是一组局部真假评分，loss 对评分取均值。训练时默认先把波形裁成 16,384 个采样点组成 batch；推理阶段只运行生成器，不运行 MPD/MRD。

### 顺着 loss 再看一次两组判别器

生成器把输入 mel 变为 $\hat x$。代码中的生成器损失可概括为：


```math

\mathcal L_G=
45\mathcal L_{\mathrm{mel}}
+\mathcal L_{\mathrm{adv}}^{\mathrm{MPD}}
+0.1\mathcal L_{\mathrm{adv}}^{\mathrm{MRD}}
+\mathcal L_{\mathrm{FM}}^{\mathrm{MPD}}
+0.1\mathcal L_{\mathrm{FM}}^{\mathrm{MRD}}.

```


- **Mel 重建损失**：比较真实和生成波形的 mel 频谱，约束内容与整体时频能量。论文简述为 mel 幅度 $L_1$；官方实现先取安全对数再做 $L_1$。
- **对抗损失**：判别器区分真假；生成器争取让 $\hat x$ 得到真实的评分。
- **特征匹配损失**：比较真实与生成音频在判别器中间层的激活，提供更稳定的多尺度反馈。

相位没有独立的角度标签损失；梯度可以沿 $p\to\cos/\sin\to\text{iSTFT}\to\hat x\to\mathcal L_G$ 回传。**架构保证输出是可做 iSTFT 的频谱形式，训练数据和损失教网络怎样填出有意义的频谱。**

## 三、额外学到的知识

```diff
- Q1：现在 neural TTS 里为什么仍看到 vocoder？它还是同一个东西吗？
```

**A：** *Vocoder* 在今天主要指**把声学表示转换成波形的职责**，内部实现可以不同。传统参数式 vocoder 利用明确的声源—滤波器等模型和 $F_0$、频谱包络等参数；神经 vocoder 从数据学习 mel → 波形或其他特征 → 波形。Tacotron 2 中的 WaveNet、HiFi-GAN、Vocos 都可以承担 vocoder 的角色，但生成机制分别不同。Vocos 是一次前向输出复数谱、再 iSTFT 的 GAN vocoder。看到该词，先问“**输入什么表示，输出什么波形**”。

```diff
- Q2：语音生成怎么分类？token、mel、flow matching 分别在哪一层？
```

**A：** 不要把它们放进同一列。至少分三个互相独立的问题：

| 维度 | 问题 | 典型选项与例子 |
|---|---|---|
| **生成对象／表示** | 模型主要生成什么？ | 连续 mel（FastSpeech 2、F5-TTS）；离散 codec／语义 token（VALL-E、MaskGCT）；直接生成波形（经典 WaveNet、DiffWave） |
| **生成规则** | 如何生成该表示？ | 自回归逐位置生成（WaveNet、一些语音 token LM）；非自回归并行预测（FastSpeech 2）；掩码迭代填充（MaskGCT）；扩散或 flow matching 对整段表示多步变换（DiffWave、F5-TTS） |
| **波形合成** | 怎样得到最终音频？ | mel → 神经 vocoder（如 Vocos）；codec token → codec decoder；直接波形建模则没有单独的转换模块 |

*Representation* 是“表示”的总称；**mel 是连续表示，codec token 是离散表示，flow matching 是生成方法**。一个系统可以混合多种步骤，例如先生成语音 token，再用 flow matching 生成连续声学特征，最后合成波形。AR/NAR 说的是生成时是否沿语音序列依赖先前输出；扩散／flow 的多轮更新不等同于逐 token 的 AR。

```diff
- Q3：flow matching 在代码里是像 Conv1d 一样的模块吗？和 diffusion 有什么区别？
```

**A：** Flow matching 是**训练目标和生成路径的设计**，不是规定网络必须使用哪种层。F5-TTS 用 DiT 预测速度；也可以在其他系统用卷积网络做预测器。用 F5-TTS 的直线路径作直观例子：真实 mel 是 $x_1$，噪声是 $x_0$，随机取 $t$，构造


```math

x_t=(1-t)x_0+tx_1,\qquad
u_t=x_1-x_0,

```


训练预测器 $v_\theta(x_t,t,\text{条件})$ 拟合方向／速度 $u_t$；推理时从噪声出发，反复预测速度，用 ODE 求解器更新整段表示，最后交给 vocoder。Diffusion 通常从前向加噪过程出发，学习预测噪声、干净样本或 score，再用反向过程采样。**两者都可能从噪声多步生成，差别在路径和预测目标；某些形式可以建立数学对应。** “多步”也不意味着沿音频时间轴一个 token 接一个 token 地自回归生成。

