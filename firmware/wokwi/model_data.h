#pragma once

// Arquivo gerado por scripts/train_and_export.py.
// Ele contem a arvore de decisao treinada e pronta para embarcar.

const int FEATURE_COUNT = 30;
const int CLASS_COUNT = 6;
const int NODE_COUNT = 61;

const char* CLASS_NAMES[CLASS_COUNT] = {
  "WALKING",
  "WALKING_UPSTAIRS",
  "WALKING_DOWNSTAIRS",
  "SITTING",
  "STANDING",
  "LAYING",
};

const char* FEATURE_NAMES[FEATURE_COUNT] = {
  "total_acc_x_mean",
  "total_acc_x_std",
  "total_acc_x_min",
  "total_acc_x_max",
  "total_acc_x_rms",
  "total_acc_y_mean",
  "total_acc_y_std",
  "total_acc_y_min",
  "total_acc_y_max",
  "total_acc_y_rms",
  "total_acc_z_mean",
  "total_acc_z_std",
  "total_acc_z_min",
  "total_acc_z_max",
  "total_acc_z_rms",
  "body_gyro_x_mean",
  "body_gyro_x_std",
  "body_gyro_x_min",
  "body_gyro_x_max",
  "body_gyro_x_rms",
  "body_gyro_y_mean",
  "body_gyro_y_std",
  "body_gyro_y_min",
  "body_gyro_y_max",
  "body_gyro_y_rms",
  "body_gyro_z_mean",
  "body_gyro_z_std",
  "body_gyro_z_min",
  "body_gyro_z_max",
  "body_gyro_z_rms",
};

const int TREE_LEFT[NODE_COUNT] = {
  1, -1, 3, 4, 5, 6, 7, -1,
  -1, 10, -1, -1, 13, 14, -1, -1,
  17, -1, -1, 20, 21, -1, -1, 24,
  25, -1, -1, 28, -1, -1, 31, 32,
  33, 34, -1, -1, 37, -1, -1, 40,
  41, -1, -1, 44, -1, -1, 47, 48,
  49, -1, -1, 52, -1, -1, 55, 56,
  -1, -1, 59, -1, -1,
};

const int TREE_RIGHT[NODE_COUNT] = {
  2, -1, 30, 19, 12, 9, 8, -1,
  -1, 11, -1, -1, 16, 15, -1, -1,
  18, -1, -1, 23, 22, -1, -1, 27,
  26, -1, -1, 29, -1, -1, 46, 39,
  36, 35, -1, -1, 38, -1, -1, 43,
  42, -1, -1, 45, -1, -1, 54, 51,
  50, -1, -1, 53, -1, -1, 58, 57,
  -1, -1, 60, -1, -1,
};

const int TREE_FEATURE[NODE_COUNT] = {
  0, -2, 1, 5, 17, 12, 0, -2,
  -2, 3, -2, -2, 14, 0, -2, -2,
  13, -2, -2, 0, 7, -2, -2, 18,
  9, -2, -2, 8, -2, -2, 3, 5,
  17, 19, -2, -2, 15, -2, -2, 15,
  15, -2, -2, 9, -2, -2, 1, 5,
  2, -2, -2, 18, -2, -2, 5, 10,
  -2, -2, 18, -2, -2,
};

const float TREE_THRESHOLD[NODE_COUNT] = {
  0.38674241f, -2.00000000f, 0.09319287f, -0.06649156f, -0.00940979f, -0.08114967f, 1.00893962f, -2.00000000f,
  -2.00000000f, 0.98457819f, -2.00000000f, -2.00000000f, 0.10223941f, 1.01321048f, -2.00000000f, -2.00000000f,
  -0.10668725f, -2.00000000f, -2.00000000f, 1.00322765f, -0.07794972f, -2.00000000f, -2.00000000f, 0.04425210f,
  0.01052554f, -2.00000000f, -2.00000000f, 0.10254675f, -2.00000000f, -2.00000000f, 1.83007401f, -0.27936275f,
  -1.10652351f, 0.51087976f, -2.00000000f, -2.00000000f, 0.06462238f, -2.00000000f, -2.00000000f, 0.11071343f,
  -0.10560709f, -2.00000000f, -2.00000000f, 0.19782069f, -2.00000000f, -2.00000000f, 0.33430274f, -0.29112686f,
  0.45041935f, -2.00000000f, -2.00000000f, 1.75168598f, -2.00000000f, -2.00000000f, -0.27038661f, -0.07960720f,
  -2.00000000f, -2.00000000f, 2.17778444f, -2.00000000f, -2.00000000f,
};

const int TREE_CLASS[NODE_COUNT] = {
  5, 5, 4, 4, 4, 4, 4, 4,
  4, 4, 4, 4, 3, 4, 4, 4,
  3, 4, 3, 3, 3, 3, 3, 3,
  3, 4, 3, 4, 4, 3, 0, 0,
  1, 1, 0, 1, 1, 1, 1, 0,
  0, 1, 0, 1, 2, 1, 2, 2,
  1, 1, 1, 2, 2, 0, 2, 2,
  2, 1, 2, 2, 2,
};
