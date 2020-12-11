## 文件说明

### 新增文件

#### README_new.md
本文件

#### knowledge/cilin.py
基于词林的分类算法主要代码。主要类为`CiLin`。

基于词林的相似性计算采用公式：

$$
sim(c1,c2) = 2*IC(LCS(c1,c2)) / (IC(c1) + IC(c2))
$$

其中LCS为最长公共前缀，比如LCS(abcd,abde)=ab

IC为information content，计算公式为：

$$
IC(c) = 1 - log(hypo(c) + 1) / log(maxnodes)
$$

其中c为一个节点，hypo为该节点下子节点数，maxnodes为全体节点数。
也就是说计算两个词的相似性等价于计算这两个词所在节点的相似性。有时候一个词可能有多种含义，这个时候相似性取最大值。

#### knowledge/data/cilin.txt
词林改进版数据

#### d.txt
“sms词表 - 词林词表”，也就是对词林来说sms中的OOV，词林对sms词表的覆盖率为75.5%

#### dd.txt
对d.txt文件中的词使用简单猜词法分类，寻找到的相似的词

简单猜词法比较暴力，比如给定一个词abc，那么将其拆成[a,b,c,ab,bc]，然后去词林中比较相似性，获得相似的IV。

可以在`knowledge/cilin.py`的main中的`test_OOV`中生成。

### 更改文件
#### build_vocab_with_class.py
新增了`class_assign_by_pos`和`class_assign_by_tree`两个函数

class_assign_by_pos直接使用POS属性进行分类，为了控制分类粒度，有一个参数`n_pos`，表示使用至多几个POS标签决定一个词的类。

class_assign_by_tree使用词林分类策略。另外，当参数`use_hownet`为True时，使用[OpenHowNet](https://github.com/thunlp/OpenHowNet/blob/master/README_ZH.md)处理OOV的问题，如果为False则用简单猜词法。

## Rescore说明

用到的文件：
* ./params/model_sm.pt:    2w词表模型
* ./params/model_4w_sm.pt: 4w词表模型
* ./data/sms/vocab2w.c.txt: 2w词表
* ./data/sms/vocab4w.c.txt: 4w词表
* ./rescore.py:            rescore程序

### `rescore.py`说明

参数：
* `--model`:    加载预训练模型路径，如`./params/model_4w_sm.pt`
* `--files_path`: 包含需要重打分的文件的目录，如`/slfs1/users/lfd98/nbest-rescore/lib/nbest/nbest-100`
* `--out_file`: 输出重打分文件路径
* `--dict`:     字典文件，需要与训练时使用的保持一致
* `--cilin`:    是否使用词林替换OOV
* `--cuda`:     -1表示cpu，其他数字表示使用哪一块gpu

实例：
``` bash
python rescore.py --modle ./params/model_4w_sm.pt -i /slfs1/users/lfd98/nbest-rescore/lib/nbest/nbest-100 -o /slfs1/users/lfd98/nbest-rescore/tmp/xxx -d data/sms/vocab4w.c.txt -c 1
```
rescore结果将会放在`slfs1/users/lfd98/nbest-rescore/tmp/xxx`下。

之前实验的结果为：

目录          | 说明                            | CER
-             | -                               | - 
bl_cls        | 2w词表类语言模型                | 15.98
bl_cls4w      | 4w词表类语言模型                | 10.97
bl_sm         | 2w词表普通模型                  | 15.10
bl_sm4w       | 4w词表普通模型                  | 10.73
bl_sm4w_cilin | 4w词表普通模型，加入词林替换OOV | 10.85
bl_sm_cilin   | 2w词表普通模型，加入词林替换OOV | 12.72


