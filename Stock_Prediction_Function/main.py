from utils import *

# 평균 제곱근 편차가 가장 낮은 값을 선택
min_model_number = 0
min_rmse = 99999
min_para = []

# 저장된 모델을 구분하기 위한 상수 지정
model_number = 1

# 각 파라미터에 대한 후보값 정의
window_size_list = [20, 50, 70, 100]
activation_type_list = ['linear', 'relu']
optimizer_type_list = ['rmsprop', 'adam']
batch_list = [10, 50, 100]

# 데이터셋 로드
df = load_dataset()

# 데이터셋 전처리 & mid 값 계산
mid_prices = preprocessing_dataset(df)

# 변수를 변경 및 적용
for window_size in window_size_list:
    for activation_type in activation_type_list:
        for optimizer_type in optimizer_type_list:
            for batch in batch_list:
                # 진행 중인 파라미터에 대한 정보 출력
                print('No. {0} === window_size : {1} === activation : {2} === optimizer : {3} === batch : {4} ==='
                      .format(model_number, window_size, activation_type, optimizer_type, batch))

                # 데이터셋 정규화
                result = normalize_dataset(mid_prices, window_size)

                # 데이터셋 분리
                x_train, x_test, y_train, y_test = split_dataset(result)

                # 모델 생성 및 훈련
                pred, rmse = build_model(model_number, window_size, activation_type, optimizer_type,
                                         batch, x_train, x_test, y_train, y_test)

                # 그래프 그리기 및 저장
                plot_graph(model_number, rmse, pred, y_test, window_size, activation_type, optimizer_type, batch)

                # 만약 기존의 편차보다 작다면
                if rmse < min_rmse:
                    min_rmse = rmse
                    min_model_number = model_number
                    del min_para[:]
                    min_para.append(window_size)
                    min_para.append(activation_type)
                    min_para.append(optimizer_type)
                    min_para.append(batch)
                model_number = model_number + 1

# 가장 낮은 편차를 가지는 모델의 파라미터 출력
print('\n############### Conclusion ###############')
print('Min Model Number : {}'.format(min_model_number))
print('Min RMSE : {}'.format(min_rmse))
print('Min Parameter : {}'.format(min_para))
