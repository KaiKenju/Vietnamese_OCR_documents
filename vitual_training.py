
import matplotlib.pyplot as plt
import re
import numpy as np

# Đường dẫn tới file train.log
log_file_path = 'train_acc2.log'

# Khởi tạo các danh sách để lưu trữ các giá trị
iters = []
train_losses = []
lrs = []
valid_losses = []
acc_full_seq = []
acc_per_char = []

# Đọc file log
with open(log_file_path, 'r') as f:
    for line in f:
        # Tìm các dòng chứa thông tin về train loss, lr
        train_match = re.match(r'iter: (\d+) - train loss: ([\d.]+) - lr: ([\de-]+)', line)
        if train_match:
            iters.append(int(train_match.group(1)))
            train_losses.append(float(train_match.group(2)))
            lrs.append(float(train_match.group(3)))

        # Tìm các dòng chứa thông tin về valid loss, acc full seq, acc per char
        valid_match = re.match(r'iter: (\d+) - valid loss: ([\d.]+) - acc full seq: ([\d.]+) - acc per char: ([\d.]+)', line)
        if valid_match:
            valid_losses.append((int(valid_match.group(1)), float(valid_match.group(2))))
            acc_full_seq.append((int(valid_match.group(1)), float(valid_match.group(3))))
            acc_per_char.append((int(valid_match.group(1)), float(valid_match.group(4))))
            

# Chuyển đổi danh sách valid_losses, acc_full_seq, acc_per_char thành các trục x, y riêng biệt
valid_iters, valid_losses = zip(*valid_losses)
_, acc_full_seq = zip(*acc_full_seq)
_, acc_per_char = zip(*acc_per_char)

# Vẽ đồ thị
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(iters, train_losses, 'b-', label='Train Loss')
plt.plot(valid_iters, valid_losses, 'r-', label='Validation Loss')

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')


plt.subplot(2, 1, 2)
plt.plot(valid_iters, acc_full_seq, 'c-', label='Accuracy Full Sequence')
plt.plot(valid_iters, acc_per_char, 'm-', label='Accuracy Per Character')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Accuracy')

# Điều chỉnh khoảng cách giữa các subplot
plt.subplots_adjust(hspace=0.5)

plt.show()
