# Google官方格式的中文版ALBERT

转换徐亮版的albert权重到Google版格式。

## 背景

徐亮的项目<a href="https://github.com/brightmart/albert_zh">albert_zh</a>训练了从tiny版到xlarge版等一系列albert模型，极大地推荐了albert在中文NLP领域的普及。

然而，徐亮版albert的开源时间早于Google版albert，这导致早期徐亮版albert的权重与Google版的不完全一致，换言之两者不能直接相互替换。当Google版开源之后，很多工作自然会以Google版为标准，但如果直接放弃掉之前训练好的权重未免就太可惜了。因此这里做一个转换。

## 说明

注意，我们说徐亮版albert跟Google版不一致，并不是单纯指变量命名上的不一致，而是模型架构上就不一致（两者处理Embedding层的方式不一样），所以原封不动的转换是做不到的。但如果放弃Embedding层的低秩分解，那么可以转换一个版本出来。

因此，本项目转换出来的模型，Embedding层都是没有低秩分解的，但是保留了transformer block的跨层参数共享。

## 权重

（上传中...）

## 交流

- QQ交流群：67729435
- 微信群请加机器人微信号spaces_ac_cn
- https://kexue.fm
