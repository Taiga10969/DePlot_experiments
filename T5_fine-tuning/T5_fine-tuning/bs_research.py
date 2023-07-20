def get_top_i(cav_file):
    top_i = {'1-gram': [], '2-gram': [], '3-gram': [], '4-gram': []}

    with open(cav_file, 'r') as file:
        lines = file.readlines()

        for line in lines:
            values = line.strip().split(', ')
            i = int(values[0])
            a, b, c, d = map(float, values[1:])

            top_i['1-gram'].append((i, a))
            top_i['2-gram'].append((i, b))
            top_i['3-gram'].append((i, c))
            top_i['4-gram'].append((i, d))

    for key in top_i:
        top_i[key] = sorted(top_i[key], key=lambda x: x[1], reverse=True)[:5]
        top_i[key] = [item[0] for item in top_i[key]]

    return top_i

cav_file = '/taiga/experiment/T5_su/record/test_bleu_record.csv'
top_i = get_top_i(cav_file)

print('Top 5 i values for 1-gram:', top_i['1-gram'])
print('Top 5 i values for 2-gram:', top_i['2-gram'])
print('Top 5 i values for 3-gram:', top_i['3-gram'])
print('Top 5 i values for 4-gram:', top_i['4-gram'])


import csv
import matplotlib.pyplot as plt

# CSVファイルのパス
csv_file = '/taiga/experiment/T5_su/record/test_bleu_record.csv'

# CSVファイルの読み込み
data = []
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append([float(value) for value in row])

# 1-gramのデータを取得
one_gram1 = [row[1] for row in data]
one_gram2 = [row[2] for row in data]
one_gram3 = [row[3] for row in data]
one_gram4 = [row[4] for row in data]

# グラフの描画
plt.figure(figsize=(10, 6))

# 点グラフの描画
#plt.scatter(one_gram1, range(1, len(one_gram1) + 1), marker='o')
#plt.scatter(one_gram2, range(1, len(one_gram2) + 1), marker='o')
plt.scatter(one_gram3, range(1, len(one_gram3) + 1), marker='o')
#plt.scatter(one_gram4, range(1, len(one_gram4) + 1), marker='o')

plt.xlabel('bleu score')
plt.ylabel('Index')
plt.title('3-gram')
plt.savefig('3-gram.png', transparent=True)
plt.savefig('3-gram.pdf')
plt.savefig('3-gram.svg', transparent=True)
plt.close()

