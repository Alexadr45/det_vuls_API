import sys
import json


def print_predict(file):
    with open(file) as f:
        data = json.load(f)
    with open('predict.txt', 'w') as f:
        for filename in data.keys():
            if data[filename] == 'Уязвимости не найдены':
                f.write('Уязвимости не найдены')
            else:
                f.writelines([f"Имя файла: {filename}\n",
                              "Уязвимые методы:\n"])
                for i, method in enumerate(data[filename].keys()):
                    f.writelines([f"Метод {i+1}:\n",
                                  "Уязвимые строки:\n",
                                  '\n'.join \
                                  (map(str,
                                       data[filename][method]['vul_lines'])),
                                  "\n", 'Код:\n',
                                  ''.join \
                                  (data[filename][method]['orig_func'])])


print_predict(sys.argv[1])
