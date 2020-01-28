# Google官方格式的中文版ALBERT

转换徐亮版的albert权重到Google版格式。

## 背景

徐亮的项目<a href="https://github.com/brightmart/albert_zh">albert_zh</a>训练了从tiny版到xlarge版等一系列albert模型，极大地推荐了albert在中文NLP领域的普及。

然而，徐亮版albert的开源时间早于Google版albert，这导致早期徐亮版albert的权重与Google版的不完全一致，换言之两者不能直接相互替换。当Google版开源之后，很多工作自然会以Google版为标准，但如果直接放弃掉之前训练好的权重未免就太可惜了。因此这里做一个转换。

## 说明

注意，我们说徐亮版albert跟Google版不一致，并不是单纯指变量命名上的不一致，而是模型架构上就不一致（两者处理Embedding层的方式不一样），所以原封不动的转换是做不到的。但如果放弃Embedding层的低秩分解，那么可以转换一个版本出来。

因此，本项目转换出来的模型，Embedding层都是没有低秩分解的，但是保留了transformer block的跨层参数共享。

## 权重

转换后的权重可以直接用<a href="https://github.com/bojone/bert4keras">bert4keras</a>加载，也可以用Google官方的<a href="https://github.com/google-research/ALBERT">albert脚本</a>加载。

|                     模型                        |           下载地址             |
|:----------------------------------------------:|:-----------------------------:|
|       albert_tiny_google_zh_489k.zip           |<a href="https://pan.baidu.com/s/1UsJRo4E8DRshwpF8rA3i9A">百度网盘</a>(4m4b)|
| albert_base_google_zh_additional_36k_steps.zip |<a href="https://pan.baidu.com/s/1QSglsiOy6cLOcSBbuHaAUQ">百度网盘</a>(tc54)|
|          albert_large_google_zh.zip            |<a href="https://pan.baidu.com/s/1YOrNYjK4oilwPLI_5e-vCw">百度网盘</a>(dq2h)|
|        albert_xlarge_google_zh_183k.zip        |<a href="https://pan.baidu.com/s/1Ny_YZ1zh2COcEdNfNXMyAg">百度网盘</a>(x89k)|

## 交流

- QQ交流群：67729435
- 微信群请加机器人微信号spaces_ac_cn
- https://kexue.fm
