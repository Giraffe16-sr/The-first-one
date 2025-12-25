import numpy as np #numpy 모듈을 np로 지칭
import matplotlib.pyplot as plt #matplotlib library를 plt로 지칭
import cv2 #cv2 모듈 불러오기
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
#가우시안 Mixture를 진행하기 위한 모듈, confusionmatrix를 그리기 위한 모듈 import

import cv2
from pathlib import Path

def fileload(
    ct1_raw_path,
    ct1_label_path,
    ct2_raw_path,
    ct2_label_path,
    ct3_raw_path,
    ct3_label_path
):
    # 파일 읽기
    ct1_raw = cv2.imread(ct1_raw_path, cv2.IMREAD_GRAYSCALE)
    ct1_target = cv2.imread(ct1_label_path, cv2.IMREAD_GRAYSCALE)

    ct2_raw = cv2.imread(ct2_raw_path, cv2.IMREAD_GRAYSCALE)
    ct2_target = cv2.imread(ct2_label_path, cv2.IMREAD_GRAYSCALE)

    ct3_raw = cv2.imread(ct3_raw_path, cv2.IMREAD_GRAYSCALE)
    ct3_target = cv2.imread(ct3_label_path, cv2.IMREAD_GRAYSCALE)

    # Label 이진화 (0 or 1)
    _, ct1_target = cv2.threshold(ct1_target, 127, 1, cv2.THRESH_BINARY)
    _, ct2_target = cv2.threshold(ct2_target, 127, 1, cv2.THRESH_BINARY)
    _, ct3_target = cv2.threshold(ct3_target, 127, 1, cv2.THRESH_BINARY)

    return ct1_raw, ct1_target, ct2_raw, ct2_target, ct3_raw, ct3_target

ct1_raw, ct1_target, ct2_raw, ct2_target, ct3_raw, ct3_target = fileload(
    r"C:\Users\surin\Desktop\data\ct1_raw.png",
    r"C:\Users\surin\Desktop\data\ct1_target.png",
    r"C:\Users\surin\Desktop\data\ct2_raw.png",
    r"C:\Users\surin\Desktop\data\ct2_target.png",
    r"C:\Users\surin\Desktop\data\ct3_raw.png",
    r"C:\Users\surin\Desktop\data\ct3_target.png",
)



# 1. CT1 출혈 ROI HU 분포 통계
roi_pixels = ct1_raw[ct1_target == 1].astype(np.float32)
mu = np.mean(roi_pixels) #mean 구하기
sigma = np.std(roi_pixels) #표준편차 구하기
print(f"[CT1 출혈 ROI] 평균 HU = {mu:.2f}, 표준편차 = {sigma:.2f}")

# 2. CT2 데이터 준비: (x, y, HU) 피처 생성
h, w = ct2_raw.shape
X, Y = np.meshgrid(np.arange(w), np.arange(h))
features = np.stack([X.ravel(), Y.ravel(), ct2_raw.ravel()], axis=1).astype(np.float32)

# 3. GMM 학습
n_components = 10 #cluster를 10으로 설정
gmm = GaussianMixture(n_components=n_components, random_state=0)
gmm.fit(features)
labels = gmm.predict(features)
label_map = labels.reshape(h, w)

from sklearn.metrics import roc_curve

# 4. 클러스터 HU 평균 확인 & 후보 선택
cluster_means = []
for i in range(n_components): #10번 반복해서 가장 출혈 부위와 유사한 클러스터를 찾음
    cluster_pixels = ct2_raw[label_map == i]
    cluster_means.append(cluster_pixels.mean())
    print(f"클러스터 {i}: 평균 HU = {cluster_pixels.mean():.2f}")

cluster_means = np.array(cluster_means)
valid_clusters = [i for i, m in enumerate(cluster_means) if 55 <= m <= 75] #출혈 부위 HU값 고려
if valid_clusters:
    dists = [abs(cluster_means[i] - mu) for i in valid_clusters]
    hemorrhage_cluster = valid_clusters[np.argmin(dists)]
    print(f"\n✅ HU 55~75 내 클러스터 {hemorrhage_cluster} 선택 (출혈 후보)")
else:
    hemorrhage_cluster = np.argmin(np.abs(cluster_means - mu))
    print(f"\n⚠️ HU 범위 만족 클러스터 없음 -> 평균에 가장 가까운 {hemorrhage_cluster} 선택")

# 5. GMM 마스크 생성
gmm_mask = (label_map == hemorrhage_cluster).astype(np.uint8)

# 6. HU 기반 Gaussian Score Map 생성 (출혈 클러스터 내부만)
score_map = np.exp(-((ct2_raw - mu) ** 2) / (2 * sigma ** 2))
score_map *= gmm_mask # GMM 마스크 내부만 고려

# 7. ROC 기반 최적 threshold 탐색
y_true = ct2_target.flatten()
y_scores = score_map.flatten()

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
optimal_idx = np.argmax(tpr - fpr) # Youden's J
optimal_threshold = thresholds[optimal_idx]
print(f"\n📊 ROC based optimal Threshold: {optimal_threshold:.4f}")

# 8. 최종 마스크 생성 (threshold 적용)
refined_mask = (score_map >= optimal_threshold).astype(np.uint8)

# 7. 후처리: 작은 잡음 제거
num_labels, labels_conn, stats, _ = cv2.connectedComponentsWithStats(refined_mask, connectivity=8)
min_area = 50 #50픽셀 이하의 점은 노이즈로 처리
clean_mask = np.zeros_like(gmm_mask)
for i in range(1, num_labels): # 0 = background
    if stats[i, cv2.CC_STAT_AREA] >= min_area:
        clean_mask[labels_conn == i] = 1

# 8. 평가
y_pred = clean_mask.flatten()
y_true = ct2_target.flatten()

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() #confusion matrix를 구해보자
accuracy = accuracy_score(y_true, y_pred)
sensitivity = recall_score(y_true, y_pred)
specificity = tn / (tn + fp)
precision = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred) #각각의 지표에 대한 수식들

print("\n📊 [GMM based confusion matrix indicators]")
print(f"Accuracy   : {accuracy:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"F1 Score   : {f1:.4f}")
print(f"ROC AUC    : {auc:.4f}")

# 9. 시각화
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("CT2 raw")
plt.imshow(ct2_raw, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("predicted (GMM + processed)")
plt.imshow(clean_mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("answer (CT2)")
plt.imshow(ct2_target, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()
