# Reading notes on the open source code of AI infrastructure

AI Infra ��ؿ�Դ������Ķ��ʼǣ��ʼǾ��Դ���ע�͵ķ�ʽ���֡�

## �ֿ����ʽ��

1. ���������Ķ�����Ĳֿ�fork��������������֧����ͬ������.

2. ���ڵ�ǰ����֧���´��룬����note-date1��֧, date1��Ӧ����֧ǰ�����һ���ύ���ڡ�

3. ��ҳ����settings�У���note-date1��֧��ΪDefault branch��

4. ��ʼ�Ķ�note-date1��֧���룬�ڴ����Ķ������н��Լ��������ע�ͷ�ʽ��ӵ������

5. ��һ��ʱ���Ķ�������֧�������ԭ�ֿ⣬��ͨ����ҳ��Sync forkֱ�ӽ�����֧���µ����£����ڸ��º������֧������һ��note-date2��

6. ��note-date2�ǻ�������֧��ԭ�ֿ���ͬ�������ģ���һ�׶ε��Ķ��ʼǻ�����note-date1�У�����Ҫʹ�öԱ������
   �ֶ���note-date1�е�����ע��Ų��note-date2�У�ͬʱ���µ��Ķ���֧��ΪDefault branch��������һ�׶ε��Ķ�����5��6�������ƽ���

7. �������Ķ���fork�ֿ���submodule�ķ�ʽ���뵽���ֿ���ͳһ����
   ��: git submodule add https://github.com/cjmcv/sglang.git sglang

## ����ʼǷ�ʽ

```bash
git clone --recursive https://github.com/cjmcv/ai-infra-reading-notes.git
# OR
git clone https://github.com/cjmcv/ai-infra-reading-notes.git
cd ai-infra-reading-notes
git submodule init
cd the-module-you-like
git pull
```