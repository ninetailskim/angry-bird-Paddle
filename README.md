# 愤怒的小鸟，体感大改造

## 林暗草惊风，将军夜引弓。平明寻白羽，没在石棱中。
![](https://ai-studio-static-online.cdn.bcebos.com/7816fb95d7c24e03aca1a4ede4d34e055c864ddd400f4b939ae038c9a169e607)

### 弓是抛射兵器中最古老的一种弹射武器。弓箭作为远射兵器，在春秋战国时期应用相当普遍，也被列为兵器之首。
### 而与之对应的“射”作为一种技艺是公卿大夫必须通晓的“六艺”之一，不仅在国君会盟、宴会上被视为一种礼仪，而且在民间风俗中也以它为礼节。

### 从传说中的后羿，到李广吕布黄忠，历史上有很多用弓好手。在现在奥林匹克运动会上，仍有射箭比赛这个项目。   
![](https://ai-studio-static-online.cdn.bcebos.com/3cb634495c8b4d8d97c4c08d001a0e664c44b97d192e49aeab3af2ab2bb5e9fa)   
### 除此之外，在影视剧中，也有射箭的相关的描写，譬如名字就点题的射雕英雄传中，郭靖、哲别都是神射手，指环王中的精灵王子莱格拉斯也是弓箭手。


### 在ACG圈子内，流传着一句话，自古弓兵多挂逼。意思是弓兵都是开挂的，除去红A和金闪闪不说，犬夜叉中的桔梗，圣斗士中的射手座，数不胜数。
![](https://ai-studio-static-online.cdn.bcebos.com/3583143bf8e346649939dd4f969da5b4939972fb6fbf45dba956d7220a1b89a3)

### 游戏里更是如此，大多数的古风或是中世纪风的游戏，都摆脱不了弓箭手这个职业。塞尔达中林克用弓不要太帅。
![](https://ai-studio-static-online.cdn.bcebos.com/5ec340c3ab964e88844424ac76b3426d54de87d1896a479fa1436d11e24313db)

## 今天我们也来练习一下射箭，我们射个大点的，射个......猪吧。
<iframe style="width:98%;height: 800px;" src="//player.bilibili.com/player.html?aid=251886520&bvid=BV1HY41147cB&cid=449098113&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>

### 我最近改写了个游戏。其实就是给愤怒的小鸟增加上了动作识别以及一个分类器。其实，愤怒的小鸟中用的弹弓也是属于弓的一种。
### 在我魔改的游戏中，我们同样要做拉弓的姿势，才能射出我们的......小鸟

### 这里的动作识别，其实就是人体关键点检测。
### 通过评估Paddlehub以及PaddleDetection中的三个关键点检测模型，我最终选择了PaddleDetection中的hrnet的模型。最终的帧率大概每秒17帧这个样子。

## 一、首先，我们要放上我们的弓箭   
### 摄像头检测到我们两只手很靠近的时候，就会把这个动作标记为我们要搭箭的状态，也就是在原游戏中通过在弹弓附近点击触发的我们要开始拉弹弓的事件。
![](https://ai-studio-static-online.cdn.bcebos.com/dd85f49aed904ff49506518ba0355ef92a1e209fba5b4a9cb1de1ab64ca831e9)   
![](https://ai-studio-static-online.cdn.bcebos.com/6c5ccf2ba88e42cc86c104c898617b8adbbc2b20e554401d9d256b8e644052cf)   
## 二、接着，我们要拉弓   
### 对，就是拉弓。然后小鸟也会对应的被拉到相应的位置。这里检测的仍然是我们的手腕，需要记录的是两个手腕的距离和相对角度。映射到游戏中，手腕的距离都是拉弓的力度，手腕的相对角度就是拉弹弓的方向。   
![](https://ai-studio-static-online.cdn.bcebos.com/6a185ef58fa140e6b324a245b7f476371cfd6a3525c34e5e8212280934331f71)   
![](https://ai-studio-static-online.cdn.bcebos.com/08a4e81c10f1428c92a957917b76c24798511dc54f2d473ea72fbfdad5d64099)   
## 三、最后，放开我们的左手，go！小鸟发射！
### 这里是一个分类模型，检测手是握拳还是手掌，握拳则为拉着弓，手掌则为放箭来了。
![](https://ai-studio-static-online.cdn.bcebos.com/8e00535f39a940ae94303794135b188655c9f145dd13419784bb439901ff7133)
![](https://ai-studio-static-online.cdn.bcebos.com/b451524e18ef45acbfced6273b6bdebc2bc88270cf314da497bf9e79dee3e553)
### 可以来看这个项目 [数据增广+手势识别[Paddlehub+PaddleX]](https://aistudio.baidu.com/aistudio/projectdetail/1923744)

# 下载模型   
链接：https://pan.baidu.com/s/1n15dxjN4h8FKl8P7nKMQLw 
提取码：bird 
### 放在src/mykeypointdetector/output_inference下
### 这个结构：src/mykeypointdetector/output_inference/hrnet_w32_384x288

# 运行
### 在src目录下，运行mymain_thread.py
### 当然，现在在AiStudio是没办法运行了，大家在本地玩一下吧~

# 个人简介

> 百度飞桨开发者技术专家 PPDE

> 飞桨上海领航团团长

> 百度飞桨官方帮帮团、答疑团成员

> 国立清华大学18届硕士

> 以前不懂事，现在只想搞钱～欢迎一起搞哈哈哈

我在AI Studio上获得至尊等级，点亮9个徽章，来互关呀！！！<br>
[https://aistudio.baidu.com/aistudio/personalcenter/thirdview/311006]( https://aistudio.baidu.com/aistudio/personalcenter/thirdview/311006)

B站ID： 玖尾妖熊

### 其他趣味项目：  
#### [Padoodle: 使用人体关键点检测让涂鸦小人动起来](https://aistudio.baidu.com/aistudio/projectdetail/2498845)
#### [利用PaddleHub制作"王大陆"滤镜](https://aistudio.baidu.com/aistudio/projectdetail/2083416)
#### [利用Paddlehub制作端午节体感小游戏](https://aistudio.baidu.com/aistudio/projectdetail/2079016)
#### [熊猫头表情生成器[Wechaty+Paddlehub]](https://aistudio.baidu.com/aistudio/projectdetail/1869462)
#### [如何变身超级赛亚人(一)--帅气的发型](https://aistudio.baidu.com/aistudio/projectdetail/1180050)
#### [【AI创造营】是极客就坚持一百秒？](https://aistudio.baidu.com/aistudio/projectdetail/1609763)    
#### [在Aistudio，每个人都可以是影流之主[飞桨PaddleSeg]](https://aistudio.baidu.com/aistudio/projectdetail/1173812)       
#### [愣着干嘛？快来使用DQN划船啊](https://aistudio.baidu.com/aistudio/projectdetail/621831)    
#### [利用PaddleSeg偷天换日～](https://aistudio.baidu.com/aistudio/projectdetail/1403330)    
