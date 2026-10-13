# 在本地下载，服务器完全离线

DOCCI官方数据入口：[Google DOCCI](https://google.github.io/docci/)。长描述和大图是新版本必须文件。DOCCI metadata155MB不是本模型必要输入，DOCCI-AAR也不需要。官方原图包约7.1GB、描述约10.5MB；Long-CLIP-B权重约599MB。带宽和解压所需空间另计。

Mac本地终端执行；已下过DOCCI则不用再下，只检查对应文件。

```bash
mkdir -p /Users/a1-6/Desktop/DOCCI
curl -fL -C - --retry 3 \
  https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines \
  -o /Users/a1-6/Desktop/DOCCI/docci_descriptions.jsonlines
curl -fL -C - --retry 3 \
  https://storage.googleapis.com/docci/data/docci_images.tar.gz \
  -o /Users/a1-6/Desktop/DOCCI/docci_images.tar.gz
mkdir -p /Users/a1-6/Desktop/LongCLIP-B
curl -fL -C - --retry 3 \
  https://huggingface.co/BeichenZhang/LongCLIP-B/resolve/main/longclip-B.pt \
  -o /Users/a1-6/Desktop/LongCLIP-B/longclip-B.pt
shasum -a 256 /Users/a1-6/Desktop/LongCLIP-B/longclip-B.pt
```

Long-CLIP官方权重SHA256必须是：

```text
ca2face3716d60473a4e284e5be2cf0a1f3a4ad9954a6a651d878e814cc0c894
```

下载有未完成文件就继续原命令断点续传，不把错误HTML当权重。已完整下载时curl可能因Range返回416；先校验哈希，不删除好文件重下。不要反序列化来源不明的`.pt`，新加载器也会先强制此官方哈希。

用你现在的SFTP/文件传输客户端上传：

- 描述和原图压缩包 → `/data1/zw/WYH/multimodel/DOCCI/`
- `longclip-B.pt` → `/data1/zw/WYH/multimodel/pretrained/longclip-B.pt`
- 代码包及checksum → `/data1/zw/WYH/multimodel/`

没有你的SSH主机地址和端口，因此这里不编造`scp`目标；使用你已有连接。源码包不含上述数据和权重。

服务器原图解压（只对尚未解压的全新目录执行）：

```bash
cd /data1/zw/WYH/multimodel
tar -tzf DOCCI/docci_images.tar.gz | head -n 12
test ! -e DOCCI/images_unpacked
mkdir DOCCI/images_unpacked
tar -xzf DOCCI/docci_images.tar.gz -C DOCCI/images_unpacked --keep-old-files
sha256sum pretrained/longclip-B.pt
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/docci_longclip_joint_v8/launch.py data-check
```

官方包只从Google官方地址获得，解压前列表应为相对图片路径，不应有`../`或绝对路径。已有images_unpacked就不要重复该段；读取器兼容嵌套目录并核对唯一basename。保持原始标注和图像，不修改/重写为COCO格式，不预先做128缩图。
