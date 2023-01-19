import http.client
import hashlib
import urllib
import random
import json
from tqdm import tqdm
import time


def translate(fromlang, tolang, seq):
    appid = '20221127001473847'  # 填写你的appid
    secretKey = 'SeNsUH0zKXFYqDQT33WZ'  # 填写你的密钥

    httpClient = None
    myurl = '/api/trans/vip/translate'

    fromLang = fromlang  # 原文语种
    toLang = tolang  # 译文语种
    salt = random.randint(32768, 65536)
    q = seq
    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    final_result = ''
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(
        q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
        salt) + '&sign=' + sign
    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)

        # response是HTTPResponse对象
        response = httpClient.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)
        final_result = result['trans_result'][0]['dst']
        # print('翻译：：')
        # print(result['trans_result'][0]['dst'])
    except Exception as e:
        print(e)
    finally:
        if httpClient:
            httpClient.close()

    return final_result






if __name__ == '__main__':
    translate_all = []
    origin = []
    print('ok')
    with open('unlabel_data.txt', 'r', encoding='UTF-8') as f:
        origin = f.readlines()
    for seq in tqdm(origin):
        time.sleep(1)
        en2fra = translate('en', 'fra', seq)
        time.sleep(1)
        fra2en = translate('fra', 'en', en2fra)
        result = fra2en + '\n'
        translate_all.append(result)
    with open('aug_unlabel_data.txt', 'w', encoding='UTF-8') as f:
        f.writelines(translate_all)
"""
import requests
import http.client
import hashlib
import urllib
import random
import json
from tqdm import tqdm
import time


class Translator:
    @staticmethod
    def translate(text, src, dst):
        src = src.replace("-", "_")
        dst = dst.replace("-", "_")
        tp = src.upper() + "2" + dst.upper()
        url = f"http://fanyi.youdao.com/translate?&doctype=json&type={tp}&i={text}"
        resp = requests.get(url)
        print(resp.content)
        return json.loads(resp.content)


def test():
    translator = Translator()
    a = translator.translate("Dude be idiots", "en", "zh-CN")
    b = a['translateResult'][0][0]['tgt']


if __name__ == '__main__':
    translate_all = []
    origin = []
    translator = Translator()
    with open('/kaggle/input/sexist/unlabel_data.txt', 'r', encoding='UTF-8') as f:
        origin = f.readlines()
    for i in range(100):
        t = i * 20
        piece = origin[t:t + 20]
        for seq in tqdm(piece):
            time.sleep(2)
            a = translator.translate(seq, "en", "zh-CN")
            en2cn = a['translateResult'][0][0]['tgt']
            time.sleep(1)
            b = translator.translate(en2cn, "zh-CN", "en")
            cn2en = b['translateResult'][0][0]['tgt']
            result = cn2en + '\n'
            translate_all.append(result)
            with open('/kaggle/working/aug_unlabel_data1.txt', 'a', encoding='UTF-8') as f:
                f.writelines(translate_all)
        time.sleep(180)"""