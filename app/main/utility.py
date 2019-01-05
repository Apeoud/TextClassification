import os
import configparser

def read_config():
    try:
        dic = dict()
        l_dic = dict()
        cf = configparser.ConfigParser()
        cf.read(os.path.dirname(os.path.dirname(__file__)) + '/config.ini', encoding='utf-8')

        for sec in cf.sections():
            prefix = ''
            if sec.endswith('_path'):
                prefix = os.path.dirname( os.path.dirname(os.path.dirname(__file__)))
            for key in cf.items(sec):
                l_dic[key[0]] = prefix + key[1]
            dic[sec] = l_dic
            l_dic = dict()
        return dic
    except Exception as e:
        print(e)
    return -1

if __name__ == "__main__":
    config = read_config()
    print(config.keys())
    print(config['keywords'].keys())
    print(config)