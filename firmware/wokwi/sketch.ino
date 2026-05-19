#include <Wire.h>
#include <math.h>
#include "model_data.h"
#include "demo_windows.h"

#define DebugSerial Serial

const int SDA_PIN = 8;
const int SCL_PIN = 9;
const int MPU6050_ADDR = 0x68;
const int SAMPLE_INTERVAL_MS = 20;

int currentSample = 0;

struct SensorReading {
  float accX;
  float accY;
  float accZ;
  float gyroX;
  float gyroY;
  float gyroZ;
  bool ok;
};

void setupMpu6050() {
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);
}

SensorReading readMpu6050() {
  SensorReading reading;
  reading.ok = false;

  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)MPU6050_ADDR, (size_t)14, true);

  if (Wire.available() < 14) {
    return reading;
  }

  int16_t ax = Wire.read() << 8 | Wire.read();
  int16_t ay = Wire.read() << 8 | Wire.read();
  int16_t az = Wire.read() << 8 | Wire.read();
  Wire.read();
  Wire.read();
  int16_t gx = Wire.read() << 8 | Wire.read();
  int16_t gy = Wire.read() << 8 | Wire.read();
  int16_t gz = Wire.read() << 8 | Wire.read();

  reading.accX = ax / 16384.0f;
  reading.accY = ay / 16384.0f;
  reading.accZ = az / 16384.0f;
  reading.gyroX = (gx / 131.0f) * 0.01745329252f;
  reading.gyroY = (gy / 131.0f) * 0.01745329252f;
  reading.gyroZ = (gz / 131.0f) * 0.01745329252f;
  reading.ok = true;
  return reading;
}

void printMpu6050Once() {
  SensorReading reading = readMpu6050();

  if (!reading.ok) {
    DebugSerial.println("MPU6050 nao respondeu nesta leitura.");
    return;
  }

  DebugSerial.print("Leitura MPU6050 | accX=");
  DebugSerial.print(reading.accX, 4);
  DebugSerial.print("g accY=");
  DebugSerial.print(reading.accY, 4);
  DebugSerial.print("g accZ=");
  DebugSerial.print(reading.accZ, 4);
  DebugSerial.print("g gyroX=");
  DebugSerial.print(reading.gyroX, 4);
  DebugSerial.print("rad/s gyroY=");
  DebugSerial.print(reading.gyroY, 4);
  DebugSerial.print("rad/s gyroZ=");
  DebugSerial.print(reading.gyroZ, 4);
  DebugSerial.println("rad/s");
}

float squareRoot(float value) {
  return sqrt(value);
}

void computeFeatures(const float window[6][WINDOW_SIZE], float features[FEATURE_COUNT]) {
  int featureIndex = 0;

  for (int sensor = 0; sensor < 6; sensor++) {
    float sum = 0.0f;
    float minValue = window[sensor][0];
    float maxValue = window[sensor][0];
    float squareSum = 0.0f;

    for (int i = 0; i < WINDOW_SIZE; i++) {
      float value = window[sensor][i];
      sum += value;
      squareSum += value * value;
      if (value < minValue) minValue = value;
      if (value > maxValue) maxValue = value;
    }

    float mean = sum / WINDOW_SIZE;
    float varianceSum = 0.0f;
    for (int i = 0; i < WINDOW_SIZE; i++) {
      float diff = window[sensor][i] - mean;
      varianceSum += diff * diff;
    }

    features[featureIndex++] = mean;
    features[featureIndex++] = squareRoot(varianceSum / WINDOW_SIZE);
    features[featureIndex++] = minValue;
    features[featureIndex++] = maxValue;
    features[featureIndex++] = squareRoot(squareSum / WINDOW_SIZE);
  }
}

int predictActivity(const float features[FEATURE_COUNT]) {
  int node = 0;

  while (TREE_LEFT[node] != -1) {
    int featureIndex = TREE_FEATURE[node];
    float threshold = TREE_THRESHOLD[node];

    if (features[featureIndex] <= threshold) {
      node = TREE_LEFT[node];
    } else {
      node = TREE_RIGHT[node];
    }
  }

  return TREE_CLASS[node];
}

void printFeatureSummary(const float features[FEATURE_COUNT]) {
  DebugSerial.println("Resumo das primeiras caracteristicas extraidas:");
  for (int i = 0; i < 6; i++) {
    DebugSerial.print("  ");
    DebugSerial.print(FEATURE_NAMES[i]);
    DebugSerial.print(" = ");
    DebugSerial.println(features[i], 6);
  }
}

void runInferenceForCurrentSample() {
  float features[FEATURE_COUNT];
  computeFeatures(DEMO_WINDOWS[currentSample], features);
  int predictedClass = predictActivity(features);
  int expectedClass = DEMO_LABELS[currentSample];

  DebugSerial.println();
  DebugSerial.println("========================================");
  DebugSerial.print("Janela simulada do dataset UCI HAR: ");
  DebugSerial.println(currentSample);
  DebugSerial.print("Classe esperada: ");
  DebugSerial.println(CLASS_NAMES[expectedClass]);
  DebugSerial.print("Classe prevista pelo ESP32-S3: ");
  DebugSerial.println(CLASS_NAMES[predictedClass]);
  DebugSerial.println(predictedClass == expectedClass ? "Resultado: acertou" : "Resultado: errou");
  printFeatureSummary(features);
  DebugSerial.println("========================================");
  DebugSerial.println("Comandos: n=proxima janela | l=inferencia live MPU6050 | r=leitura crua");
}

bool collectLiveWindow(float window[6][WINDOW_SIZE]) {
  for (int i = 0; i < WINDOW_SIZE; i++) {
    SensorReading reading = readMpu6050();

    if (!reading.ok) {
      return false;
    }

    window[0][i] = reading.accX;
    window[1][i] = reading.accY;
    window[2][i] = reading.accZ;
    window[3][i] = reading.gyroX;
    window[4][i] = reading.gyroY;
    window[5][i] = reading.gyroZ;
    delay(SAMPLE_INTERVAL_MS);
  }

  return true;
}

void runLiveInferenceFromMpu6050() {
  float liveWindow[6][WINDOW_SIZE];
  float features[FEATURE_COUNT];

  DebugSerial.println();
  DebugSerial.println("Coletando 128 leituras do MPU6050 a aproximadamente 50 Hz...");

  if (!collectLiveWindow(liveWindow)) {
    DebugSerial.println("Falha: MPU6050 nao respondeu durante a coleta.");
    return;
  }

  computeFeatures(liveWindow, features);
  int predictedClass = predictActivity(features);

  DebugSerial.println();
  DebugSerial.println("========================================");
  DebugSerial.println("Inferencia live usando leituras do MPU6050");
  DebugSerial.print("Classe prevista pelo ESP32-S3: ");
  DebugSerial.println(CLASS_NAMES[predictedClass]);
  DebugSerial.println("Observacao: no Wokwi, o sensor fica praticamente parado se voce nao alterar os valores.");
  printFeatureSummary(features);
  DebugSerial.println("========================================");
  DebugSerial.println("Comandos: n=proxima janela | l=inferencia live MPU6050 | r=leitura crua");
}

void setup() {
  DebugSerial.begin(115200);
  delay(500);

  DebugSerial.println();
  DebugSerial.println("BOOT OK - Serial USB ativa em 115200");
  DebugSerial.println("========================================");
  DebugSerial.println("Projeto Final - IA Embarcada e Modelos Compactos");
  DebugSerial.println("Classificador de atividade humana com UCI HAR + ESP32-S3");
  DebugSerial.println("Modos disponiveis:");
  DebugSerial.println("  n = testar proxima janela real do dataset UCI HAR");
  DebugSerial.println("  l = coletar 128 leituras do MPU6050 e inferir no ESP32-S3");
  DebugSerial.println("  r = mostrar uma leitura instantanea do MPU6050");
  DebugSerial.println("========================================");

  setupMpu6050();
  DebugSerial.println("MPU6050 inicializado.");
  printMpu6050Once();
  runInferenceForCurrentSample();
}

void loop() {
  if (DebugSerial.available()) {
    char command = DebugSerial.read();
    if (command == 'n' || command == 'N') {
      currentSample = (currentSample + 1) % DEMO_SAMPLE_COUNT;
      runInferenceForCurrentSample();
    } else if (command == 'l' || command == 'L') {
      runLiveInferenceFromMpu6050();
    } else if (command == 'r' || command == 'R') {
      printMpu6050Once();
    }
  }
}
