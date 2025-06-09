import pandas as pd

# ① 데이터 로드
train_path = "./data/train.csv"
df = pd.read_csv(train_path, low_memory=False)

# ② 데이터에 존재하는 몰드코드 자동 추출
mold_codes = df['mold_code'].unique()

# ③ 샘플링
sampled_list = []

for mold in mold_codes:
    df_mold = df[df['mold_code'] == mold]
    sampled = df_mold.tail(200)  # 각 몰드코드별 마지막 200개 (200개 미만이면 전부)
    sampled_list.append(sampled)

# ④ 전부 합치기
sampled_df = pd.concat(sampled_list, ignore_index=True)

# ⑤ 저장
sampled_df.to_csv("./data/sampled_train.csv", index=False)

print("샘플링 완료: sampled_train.csv 저장됨")

# 몰드코드별 샘플 개수 확인
mold_counts = sampled_df['mold_code'].value_counts().sort_index()

print(mold_counts)
