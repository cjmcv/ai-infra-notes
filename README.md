# Reading notes on open source code of AI infrastructure (LLM, Inference, HPC)

AI Infra ��ؿ�Դ������Ķ��ʼǣ��ʼ���Ҫ�Դ���ע�͵ķ�ʽ���� (�� `<NT>` ���)�����������ݻ�����ĵ��ʼ��Ը���Դ���Ķ���

## �ʼ��б�-Դ��ע��

| ��Ŀ�� | �ʼǷ�Χ | �ʼ��� | ��Ҫϵ�� | ���������� |
| :---: | :--- | :--- | :---: | :---: |
| sglang | ���徫�� | 380 | :star::star::star: | 20250608 | 
| cutlass | �����Ķ� | 51 | :star::star::star: | 20250620 |
| flash-attention | /hopper/ | 94 | :star::star: | 20250620 |
| lighteval | metrics����ָ�겿�� | 4 |:star: | 20250210 |

## �ĵ�/ͼƬ�ʼ�-docs

| ��Ŀ�� | ���� | ���������� |
| :---: | :--- | :---: | 
| sglang | [kvcache��������](https://github.com/cjmcv/ai-infra-notes/tree/master/docs/sglang/kvcache) | 20250326 |
| sglang | [�����ṹ](https://github.com/cjmcv/ai-infra-notes/tree/master/docs/sglang/sglang-architecture-20250117.jpg) | 20250117 |
| flash-attention | [��������ṹ](https://github.com/cjmcv/ai-infra-notes/tree/master/docs/flash-attention/flash-attention-code-structure-20250620.png) | 20250620 |
| flash-attention | [����ģ���ϵ](https://github.com/cjmcv/ai-infra-notes/tree/master/docs/flash-attention/flash-attention-module-relationship-20250620.png) | 20250620 |
| lighteval | [����ָ������](https://github.com/cjmcv/ai-infra-notes/tree/master/docs/lighteval/lighteval-metrics-20250218.jpg) | 20250218 |

## ����ʼǷ�ʽ

```bash
git clone https://github.com/cjmcv/ai-infra-notes.git
cd ai-infra-notes
# ��ȡ�����Ȥ��ģ���Ķ�����sglangΪ��
git submodule init sglang
git submodule update sglang
```

�ʼǲ��ֻ��� `<NT>` ��ǣ����������鿴

* ע�⣺�緢��������ʾ���룬�ɳ��԰�GB2312��UTF8��ʽ�鿴����vscodeΪ����������½ǵı����ʽ��ѡReopen with Encoding���л���GB2312��UTF8�鿴��

## �ֿ����ʽ

1. ���������Ķ�����Ĳֿ�fork��������������֧����ͬ������.

2. ���ڵ�ǰ����֧���´��룬����note-date1��֧, date1��Ӧ����֧ǰ�����һ���ύ���ڡ�

3. ��ҳ����settings�У���note-date1��֧��ΪDefault branch��

4. ��ʼ�Ķ�note-date1��֧���룬�ڴ����Ķ������н��Լ��������ע�ͷ�ʽ��ӵ������

5. ��һ��ʱ���Ķ�������֧�������ԭ�ֿ⣬��ͨ����ҳ��Sync forkֱ�ӽ�����֧���µ����£����ڸ��º������֧������һ��note-date2��

6. ��note-date2�ǻ�������֧��ԭ�ֿ���ͬ�������ģ���һ�׶ε��Ķ��ʼǻ�����note-date1�У�����Ҫʹ�öԱ�������ֶ���note-date1�е�����ע��Ų��note-date2�У�ͬʱ���µ��Ķ���֧��ΪDefault branch��������һ�׶ε��Ķ�����5��6�������ƽ���

7. �������Ķ���fork�ֿ���submodule�ķ�ʽ���뵽���ֿ���ͳһ����

```bash
git submodule add https://github.com/cjmcv/sglang.git sglang
```