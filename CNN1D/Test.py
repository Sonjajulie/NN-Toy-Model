from sklearn.preprocessing import StandardScaler
import numpy as np
pause(5000)
data =np.array([[2,7,0,7,4,3,9,1,8,4,10,6,4,5,2,3,5,8,3,4],
            [10,6,7,10,8,1,3,4,4,1,1,7,10,8,0,0,2,4,6,9]]);
dataT=data.T
data_mean=np.mean(data,axis=1)
len_x,len_y = data.shape
sigma_var = np.std(data,axis=1)
data_standardized=[(data[i]-data_mean[i])/sigma_var[i] for i in range(len(sigma_var))]
#
print(data_standardized)
# print(np.mean(data_standardized,axis=1))
# print(np.std(data_standardized))
# standardized_data=[(data[i]-data_mean[i])/sigma_var[i] for i in range(len(data))]
# print(standardized_data)

#
#
# #[number_of_samples, number_of_features]
# #
# scaler = StandardScaler()
# print(scaler.fit(dataT))
# # print(scaler.mean_)
# # print(scaler.mean_==np.mean(dataT,axis=0))
# trans_data=scaler.transform(dataT)
# print(trans_data.T)
#
# # print(scaler.transform(data))
# # print(scaler.transform([[2, 2]]))


# a=np.array([
# [[0, 1, 2],
# [3, 4, 5]],
# [[0, 1.1, 2.2],
# [3.3, 4.4, 5.5]],
# [[0, 1.11, 2.22],
# [3.33, 4.44, 5.55]],
#     [[0, 1.111, 2.222],
#      [3.333, 4.444, 5.555]]])
# print(a.shape)
# print(a)
# print(np.reshape(a, (12,2), order='C').T)
