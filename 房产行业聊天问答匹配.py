import paddle
import pandas as pd
import numpy as np 
import random

paddle.seed(2022)
np.random.seed(2022)
random.seed(2022)

# 自定义数据读取函数
def read_data(query_data_file, reply_data_file, is_test=False):
    '''
    :param query_data_file: query文件路径
    :param reply_data_file: reply文件路径
    :param is_test: 如果读取训练集或验证集数据，则设为False，测试集设为True
    :return:
    '''
    # 使用pandas读取query文件
    df_query = pd.read_csv(query_data_file, sep='\t', names=['query_id', 'query_string'])
    # 读取reply文件
    if not is_test:
        df_reply = pd.read_csv(reply_data_file, sep='\t', names=['query_id', 'reply_id', 'reply_string', 'label'])
    else:
        df_reply = pd.read_csv(reply_data_file, sep='\t', names=['query_id', 'reply_id', 'reply_string'])
    df = pd.merge(df_query, df_reply, on='query_id')  # 合并query和reply，数据增强
    df = df.dropna(axis=0, how='any')  # 删除缺失值
    for index, row in df.iterrows():
        if not is_test:
            if row["label"] == 1:
                for i in range(2):
                    yield {
                    "query_string": row["query_string"],
                    "reply_string": row["reply_string"],
                    "label": row["label"]
                }
            yield {
                "query_string": row["query_string"],
                "reply_string": row["reply_string"],
                "label": row["label"]
            }
        else:
            yield {
                "query_id": row["query_id"],
                "reply_id": row["reply_id"],
                "query_string": row["query_string"],
                "reply_string": row["reply_string"],
            }

from paddlenlp.datasets import load_dataset, MapDataset
from sklearn.model_selection import train_test_split

# 加载训练集
dataset = load_dataset(read_data,
                             query_data_file='work/data/train/train.query.tsv',
                             reply_data_file='work/data/train/train.reply.tsv',
                             is_test=False,
                             lazy=False)

# 按照0.1的比例随机划分验证集
all_data = dataset.data
train_data, dev_data = train_test_split(all_data, test_size=0.1)
train_dataset = MapDataset(data=train_data)
dev_dataset = MapDataset(data=dev_data)

from paddlenlp.transformers import ErnieTokenizer

MODEL_NAME = "ernie-2.0-base-zh"
# 定义tokenizer
tokenizer = ErnieTokenizer.from_pretrained(MODEL_NAME)

from functools import partial
from paddlenlp.transformers import ErnieTokenizer

# 设置模型的最大序列长度为128
def convert_example(data, tokenizer, max_seq_len=128, is_test=False):  
    # 调用tokenizer的数据处理方法将文本转为id
    inputs = tokenizer(text=data["query_string"],
                       text_pair=data["reply_string"],
                       max_seq_len=max_seq_len)
    if not is_test:
        return {
            "input_ids": inputs["input_ids"],
            # 标记输出的id属于哪个text,为0属于query_string,为1属于reply_string
            "token_type_ids": inputs["token_type_ids"],  
            "label": data["label"]
        }
    else:
        return {
            "query_id": data["query_id"],
            "reply_id": data["reply_id"],
            "input_ids": inputs["input_ids"],
            "token_type_ids": inputs["token_type_ids"]
        }

# 数据转换函数
train_convert_func = partial(
    convert_example, 
    tokenizer=tokenizer, 
    max_seq_len=128, 
    is_test=False)

# 数据集的每个数据作为train_convert_func函数的参数运行输出
# batched=False表示转换操作作用于单条样本
# lazy=False表示转换的操作延迟到实际取数据时进行
train_dataset.map(train_convert_func, batched=False, lazy=False)
dev_dataset.map(train_convert_func, batched=False, lazy=False)

from paddlenlp.data import Dict, Pad, Stack
from paddle.io import DataLoader, BatchSampler

# 设置大小合适的batch_size，即一次训练所抓取的数据样本数量
batch_size = 32

# 初始化BatchSampler取样器，将分词id化的数据按照batch_size分组
train_batch_sampler = BatchSampler(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True)
dev_batch_sampler = BatchSampler(
    dev_dataset,
     batch_size=batch_size, 
     shuffle=False)

# 定义batchify_fn，创建一个字典，对数据进行批量打包+pad
# lambda表达式用于构造匿名函数
batchify_fn = lambda example, fn=Dict({
    "input_ids": Pad(pad_val=tokenizer.pad_token_id, dtype="int64"),
    "token_type_ids": Pad(pad_val=tokenizer.pad_token_type_id, dtype="int64"),
    "label": Stack(axis=0, dtype='int64')
}): fn(example)
# 将长度不同的多个句子padding到统一长度，取N个输入数据中的最大长度
# 长度是指的： 一个batch中的最大长度，主要考虑性能开销

# 初始化DataLoader数据加载器，指定sampler并通过batch_sampler分出batch
# collate_fn将batch数据重新组装成bacthify_fn形式
train_dataloader = DataLoader(
    train_dataset, 
    batch_sampler=train_batch_sampler, 
    collate_fn=batchify_fn)
dev_dataloader = DataLoader(
    dev_dataset, 
    batch_sampler=dev_batch_sampler,
     collate_fn=batchify_fn)

from paddlenlp.transformers import ErnieForSequenceClassification

# 定义模型
model = ErnieForSequenceClassification.from_pretrained(MODEL_NAME, num_classes=2)

# fgm对抗训练
class FGM():
    """针对embedding层梯度上升干扰的对抗训练方法,Fast Gradient Method（FGM）"""

    def __init__(self, model, epsilon=1.):
        self.model = model  # self为实例对象，相当于this
        self.epsilon = epsilon  # 最小正浮点数
        self.backup = {}  # 备份

    def attack(self,  emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding词嵌入的参数名
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and emb_name in name:  # 检验参数是否可训练及范围
                self.backup[name] = param.numpy()  # 备份原有参数值
                grad_tensor = paddle.to_tensor(param.grad)  # param.grad是个numpy对象
                norm = paddle.norm(grad_tensor)  # norm化
                if norm != 0:
                    r_at = self.epsilon * grad_tensor / norm
                    param.add(r_at)  # 在原有embed值上添加向上梯度干扰

    def restore(self, emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding词嵌入的参数名
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and emb_name in name:
                assert name in self.backup
                param.set_value(self.backup[name])  # 将原有embed参数还原
        self.backup = {}

import paddle
from paddlenlp.transformers import LinearDecayWithWarmup

# 设置学习率优化策略
num_epoch = 5  # 模型训练迭代总次数
learning_rate = 1e-5  # 学习率，控制模型学习进度
warm_step = 0.1  # 预热后学习率
grad_norm = 1.0  # 允许最大值，小于等于则数据不裁剪
weight_decay = 0.01  # 权值衰减，调节模型复杂度对损失函数的正比影响，防止过拟合

# 定义lr scheduler
# 创建学习率计划程序，该调度程序线性增加学习率 从 0 到给定，
# 在此预热期之后学习率 将从基本学习率线性降低到 0。
scheduler = LinearDecayWithWarmup(
    learning_rate=learning_rate,
    total_steps=len(train_dataloader)*num_epoch,  # 训练步骤数
    warmup=warm_step
)

# 衰减的参数
# 迭代打印model.named_parameters()将会打印每一次迭代元素的名字和param。
decay_params = [
    p.name for n, p in model.named_parameters()
    if not any([nd in n for nd in ["bias", "norm"]])
]
# 定义优化器，更新和计算影响模型训练和模型输出的网络参数，使其逼近或达到最优值，从而最小化损失函数
optimizer = paddle.optimizer.AdamW(parameters=model.parameters(),
                       learning_rate=scheduler,
                       weight_decay=weight_decay,
                       apply_decay_param_fun=lambda x: x in decay_params,
                       grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=grad_norm))

import numpy as np
from paddle.metric import Metric

# 判断数据类型，若var是np.ndarray, np.generic类型返回true,反之为false
def _is_numpy_(var):
    return isinstance(var, (np.ndarray, np.generic))

# 计算F1值，准确率和召回率的调和平均值
class F1Score(Metric):

    def __init__(self, name='f1score', *args, **kwargs):
        super(F1Score, self).__init__(*args, **kwargs)
        self.tp = 0  # 正例预测正确的个数
        self.fp = 0  # 负例预测错误的个数
        self.fn = 0  # 正例预测错误的个数
        self.tn = 0  # 负例预测正确的个数
        self._name = name

    def update(self, preds, labels):

        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        elif not _is_numpy_(preds):
            raise ValueError("The 'preds' must be a numpy ndarray or Tensor.")

        if isinstance(labels, paddle.Tensor):
            labels = labels.numpy()
        elif not _is_numpy_(labels):
            raise ValueError("The 'labels' must be a numpy ndarray or Tensor.")

        sample_num = labels.shape[0]
        preds = np.argmax(preds, axis=-1)

        for i in range(sample_num):
            pred = preds[i]
            label = labels[i]
            if pred == 1:
                if pred == label:
                    self.tp += 1
                else:
                    self.fp += 1
            else:
                if label == 1:
                    self.tn += 1
                else:
                    self.fn += 1

    def reset(self):
        """
        Resets all of the metric state.
        """
        self.tp = 0  # 正例预测正确的个数
        self.fp = 0  # 负例预测错误的个数
        self.fn = 0  # 正例预测错误的个数
        self.tn = 0  # 负例预测正确的个数

    def accumulate(self):
        ap = self.tp + self.fp
        precision_ =  float(self.tp) / ap if ap != 0 else .0
        ad = self.tp + self.tn
        recall_ = float(self.tp) / ad if ad != 0 else .0
        return 2*(precision_*recall_) / (precision_+recall_)

    def name(self):
        return self._name

from paddle.metric import Accuracy
import paddle.nn.functional as F
from loguru import logger

# 定义评估函数
@paddle.no_grad()
def evaluation():
    # 计算loss、accuracy、f1值
    model.eval()
    avg_loss = 0.
    acc_metric = Accuracy()  # 准确率：分类正确与样本总数之比
    f1_metric = F1Score()
    for batch in dev_dataloader:
        input_ids, token_type_id, label = batch
        logit = model(input_ids=input_ids, token_type_ids=token_type_id)
        loss = loss_function(logit, label.squeeze())
        avg_loss += loss.item()
        result = acc_metric.compute(logit, label)
        acc_metric.update(result)
        f1_metric.update(F.softmax(logit), label)
    f1 = f1_metric.accumulate()
    acc = acc_metric.accumulate()
    model.train()
    return avg_loss / len(dev_dataloader), acc, f1

log_step = 50
eval_step = 200

# 定义损失函数
loss_function = paddle.nn.CrossEntropyLoss()

# 开始训练
global_step = 0
fgm = FGM(model)
for epoch in range(num_epoch):
    model.train()
    for batch_id, batch in enumerate(train_dataloader):
        input_ids, token_type_id, label = batch
        # 正常训练得到前向loss
        logit = model(input_ids=input_ids, token_type_ids=token_type_id)
        loss = loss_function(logit, label)

        # loss反传传播，计算梯度
        loss.backward()
        # fgm对抗训练
        fgm.attack()  # 在embedding上添加对抗扰动
        logit_adv = model(input_ids=input_ids, token_type_ids=token_type_id)
        loss_adv = loss_function(logit_adv, label)
        loss_adv.backward()  # 反向传播，在正常梯度基础上，累加对抗训练得梯度
        fgm.restore()  # 恢复embedding参数
        # 梯度下降，更新参数
        optimizer.step()
        scheduler.step()
        optimizer.clear_grad()

        global_step += 1
        if global_step % log_step == 0:
            logger.info("epoch: {}, batch: {}, global_step: {}, loss: {}".format(epoch,
            batch_id, global_step, loss.item()))
        if global_step % eval_step == 0:
            dev_loss, acc, f1 = evaluation()
            logger.info("global_step: {}, dev_loss: {}, acc: {}, f1: {}".format(
                global_step, dev_loss, acc, f1
            ))

# 训练完成后在进行一次评估
dev_loss, acc, f1 = evaluation()
logger.info("global_step: {}, dev_loss: {}, acc: {}, f1: {}".format(
    global_step, dev_loss, acc, f1
))

# 加载测试集，使用训练好的模型参数在测试集上进行预测，得到预测结果。
test_dataset = load_dataset(read_data,
                             query_data_file='work/data/test/test.query.tsv',
                             reply_data_file='work/data/test/test.reply.tsv',
                             is_test=True,
                             lazy=False)
print(test_dataset[0])
from tqdm import tqdm

# 生成提交文件
with open("submission.tsv", "wt", encoding="utf-8") as f:
    for data in tqdm(test_dataset):
        inputs = tokenizer(text=data["query_string"], text_pair=data["reply_string"], max_seq_len=128)
        inputs = {k: paddle.to_tensor(v).unsqueeze(axis=0) for k, v in inputs.items()}
        logit = model(**inputs)
        result = paddle.argmax(logit, axis=-1).item()
        f.write(str(data["query_id"]) + '\t' + str(data["reply_id"]) + '\t' + str(result) + '\n')