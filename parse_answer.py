import sys
import json


def print_predict(file):
    with open(file) as f:
        data = json.load(f)
    with open('predict.txt', 'w') as f:
        for name in data.keys():
            if data[filename] == 'Уязвимости не найдены':
                f.write('Уязвимости не найдены')
            else:
                f.writelines([f"Имя файла: {name}\n",
                              "Уязвимые методы:\n"])
                for i, m in enumerate(data[name].keys()):
                    f.writelines([f"Метод {i+1}:\n",
                                  "Уязвимые строки:\n",
                                  '\n'.join(map(str,
                                       data[name][m]['vul_lines'])),
                                  "\n", 'Код:\n',
                                  ''.join(data[name][m]['orig_func'])])


print_predict(sys.argv[1])
