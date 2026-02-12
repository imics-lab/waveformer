from tsai.basics import *
from fastai.callback.core import *
from fastai.callback.tracker import EarlyStoppingCallback

# ─── 1. Convert tensors to numpy and transpose to tsai format ─
# Your dataloader: (samples, timesteps, channels)
# tsai expects:    (samples, channels, timesteps)
X_train_np = X_train.numpy().transpose(0, 2, 1)
X_valid_np = X_valid.numpy().transpose(0, 2, 1)
X_test_np  = X_test.numpy().transpose(0, 2, 1)
y_train_np = y_train.numpy()
y_valid_np = y_valid.numpy()
y_test_np  = y_test.numpy()
 
# ─── 2. Concatenate train+valid into one array, build splits ──
# This is exactly how tsai's own get_classification_data works
X     = np.concatenate([X_train_np, X_valid_np], axis=0)
y     = np.concatenate([y_train_np, y_valid_np], axis=0)
splits = (list(range(len(X_train_np))),
          list(range(len(X_train_np), len(X))))
 
# ─── 3. Train ─────────────────────────────────────────────────
tfms       = [None, TSClassification()]
batch_tfms = TSStandardize(by_sample=True)
 
mv_clf = TSClassifier(X, y, splits=splits, path='.', arch="TSTPlus",
                      tfms=tfms, batch_tfms=batch_tfms, metrics=accuracy)
mv_clf.remove_cbs(mv_clf.progress)   # fixes Colab NBMasterBar crash
mv_clf.fit_one_cycle(50, 1e-3)
 
# ─── 4. Predict on test set ───────────────────────────────────
import torch
 
mv_clf.model.eval()
with torch.no_grad():
    # Convert test data to tensor in the same format the model expects
    x_test_tensor = torch.from_numpy(X_test_np).float()
    # Run in small batches to avoid memory issues
    all_preds = []
    bs = 64
    for i in range(0, len(x_test_tensor), bs):
        batch = x_test_tensor[i:i+bs].to(mv_clf.dls.device)
        out = mv_clf.model(batch)
        all_preds.append(out.cpu())
    preds = torch.cat(all_preds, dim=0)
 
preds_labels  = preds.argmax(dim=1).numpy()
target_labels = y_test_np
 
# ─── 5. Report ────────────────────────────────────────────────
t_names = ['longblink', 'shortblink']
 
cm    = confusion_matrix(target_labels, preds_labels)
cm_df = pd.DataFrame(cm, index=t_names, columns=t_names)
 
fig = plt.figure(figsize=(6.5, 5))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='cubehelix_r')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'InceptionTimePlus using SelfRegulationSCP2\nAccuracy:{accuracy_score(target_labels, preds_labels):.3f}')
plt.show()
 
print(f'Prediction accuracy: {accuracy_score(target_labels, preds_labels):.3f}')
print(classification_report(target_labels, preds_labels, target_names=t_names))