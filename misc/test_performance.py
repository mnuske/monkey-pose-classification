# gt.shape = 600
# pred.shape = 600
from sklearn.metrics import confusion_matrix
import numpy as np
class_names = ['sitting', 'standing2legs', 'standing4legs', 'walking']
gt = np.load('gt.npy')
pred = np.load('pred.npy')
cmat = confusion_matrix(gt, pred)
import matplotlib.pyplot as plt
plt.imshow(cmat, cmap='plasma')
plt.colorbar()
plt.title('Confusion Matrix of Classification', fontsize=20)
plt.xlabel('Prediction', labelpad=10, fontsize=20)
plt.ylabel('Ground-Truth', labelpad=10, fontsize=20)
plt.xticks(np.arange(len(class_names)), class_names)
plt.xticks(rotation=25)
plt.yticks(np.arange(len(class_names)), class_names)
plt.tight_layout()
plt.show()