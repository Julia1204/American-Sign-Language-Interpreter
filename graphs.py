import numpy as np
from matplotlib import pyplot as plt

npy_file_path = 'Dataall/thanks/22/8.npy'


pose_start = 0
pose_end = pose_start + (33 * 4)
face_start = pose_end
face_end = face_start + (468 * 3)
left_hand_start = face_end
left_hand_end = left_hand_start + (21 * 3)
right_hand_start = left_hand_end
right_hand_end = right_hand_start + (21 * 3)

loaded_data = np.load(npy_file_path)

print(loaded_data)
print(loaded_data[3])
plt.plot(np.arange(left_hand_start, left_hand_end), loaded_data[left_hand_start:left_hand_end], label='Punkty lewej ręki', color='red')
plt.title('Wizualizacja punktów lewej ręki')
plt.xlabel('Indeks wektora')
plt.ylabel('Wartość')
plt.legend()
plt.show()

plt.plot(np.arange(right_hand_start, right_hand_end), loaded_data[right_hand_start:right_hand_end], label='Punkty prawej ręki', color='orange')
plt.title('Wizualizacja punktów prawej ręki')
plt.xlabel('Indeks wektora')
plt.ylabel('Wartość')
plt.legend()
plt.show()

# histogram
plt.hist(loaded_data[left_hand_start:left_hand_end], bins=50, label='Punkty lewej ręki', color='red', alpha=0.7)
plt.hist(loaded_data[right_hand_start:right_hand_end], bins=50, label='Punkty prawej ręki', color='orange', alpha=0.7)
plt.title('Histogram punktów rąk')
plt.xlabel('Wartość punktów')
plt.ylabel('Liczebność')
plt.legend()
plt.show()

# Both hands
plt.plot(loaded_data[left_hand_start:left_hand_end], label='Punkty lewej ręki', color='red')
plt.plot(loaded_data[right_hand_start:right_hand_end], label='Punkty prawej ręki', color='orange')
plt.title('Wizualizacja punktów rąk w czasie')
plt.xlabel('Czas (indeks wektora)')
plt.ylabel('Wartość')
plt.legend()
plt.show()


left_hand_points = loaded_data[left_hand_start:left_hand_end].reshape(-1, 3)
right_hand_points = loaded_data[right_hand_start:right_hand_end].reshape(-1, 3)
print(left_hand_points)
print(right_hand_points)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(left_hand_points[:, 0], left_hand_points[:, 1], left_hand_points[:, 2], label='Punkty lewej ręki', color='red')
ax.scatter(right_hand_points[:, 0], right_hand_points[:, 1], right_hand_points[:, 2], label='Punkty prawej ręki', color='orange')
ax.set_title('Wizualizacja punktów rąk w przestrzeni 3D')
ax.set_xlabel('Oś X')
ax.set_ylabel('Oś Y')
ax.set_zlabel('Oś Z')
ax.xaxis.labelpad = 10
ax.yaxis.labelpad = 10
ax.zaxis.labelpad = 10
ax.legend()
plt.show()


pose_data_reshaped = loaded_data[pose_start:pose_end].reshape(-1, 4)
print(pose_data_reshaped)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x_values = pose_data_reshaped[:, 0]
y_values = pose_data_reshaped[:, 1]
z_values = pose_data_reshaped[:, 2]
visibility_values = pose_data_reshaped[:, 3]
scaled_sizes = 50 * visibility_values
ax.scatter(x_values, y_values, z_values, label='Punkty pozy', color='red', alpha=0.7, s=scaled_sizes)
ax.set_title('Wizualizacja punktów pozy w przestrzeni 3D z widocznością')
ax.set_xlabel('Oś X')
ax.set_ylabel('Oś Y')
ax.set_zlabel('Oś Z')
ax.xaxis.labelpad = 10
ax.yaxis.labelpad = 10
ax.zaxis.labelpad = 10
ax.legend()
plt.show()