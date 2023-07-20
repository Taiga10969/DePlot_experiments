import csv
import matplotlib.pyplot as plt

# CSVファイルのパス
csv_file = '/taiga/experiment/T5_su/record/loss_record.csv'

# データを格納するリスト
epochs = []
train_losses = []
val_losses = []

# CSVファイルからデータを読み取る
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # ヘッダ行をスキップ
    for row in reader:
        epoch, train_loss, val_loss = row
        epochs.append(int(epoch))
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))

# 学習曲線のプロット
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.savefig('loss.png')
plt.savefig('loss.svg')
plt.close()