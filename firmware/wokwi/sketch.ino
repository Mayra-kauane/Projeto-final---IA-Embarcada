#include <Wire.h>
#include <math.h>
#include "model_data.h"
#include "demo_windows.h"

#define DebugSerial Serial

const int SDA_PIN = 8;
const int SCL_PIN = 9;
const int MPU6050_ADDR = 0x68;
const int SAMPLE_INTERVAL_MS = 20;
const int LIVE_INFERENCE_INTERVAL_MS = 2500;

int currentSample = 0;
bool liveMonitorEnabled = true;
bool liveWindowReady = false;
int liveSampleIndex = 0;
unsigned long lastLiveSampleMs = 0;
unsigned long lastLiveInferenceMs = 0;
float liveWindow[6][WINDOW_SIZE];

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

void chooseRandomDemoSample() {
  if (DEMO_SAMPLE_COUNT <= 1) {
    currentSample = 0;
    return;
  }

  int nextSample = random(DEMO_SAMPLE_COUNT - 1);
  if (nextSample >= currentSample) {
    nextSample++;
  }
  currentSample = nextSample;
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
  float hidden[HIDDEN_COUNT];

  for (int neuron = 0; neuron < HIDDEN_COUNT; neuron++) {
    float sum = MLP_B1[neuron];

    for (int feature = 0; feature < FEATURE_COUNT; feature++) {
      float scale = SCALER_SCALE[feature];
      float normalized = scale == 0.0f ? 0.0f : (features[feature] - SCALER_MEAN[feature]) / scale;
      sum += normalized * MLP_W1[feature][neuron];
    }

    hidden[neuron] = sum > 0.0f ? sum : 0.0f;
  }

  int bestClass = 0;
  float bestScore = MLP_B2[0];

  for (int neuron = 0; neuron < HIDDEN_COUNT; neuron++) {
    bestScore += hidden[neuron] * MLP_W2[neuron][0];
  }

  for (int classIndex = 1; classIndex < CLASS_COUNT; classIndex++) {
    float score = MLP_B2[classIndex];

    for (int neuron = 0; neuron < HIDDEN_COUNT; neuron++) {
      score += hidden[neuron] * MLP_W2[neuron][classIndex];
    }

    if (score > bestScore) {
      bestScore = score;
      bestClass = classIndex;
    }
  }

  return bestClass;
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

void runInferenceForWindow(const float window[6][WINDOW_SIZE], const char* title) {
  float features[FEATURE_COUNT];
  computeFeatures(window, features);
  int predictedClass = predictActivity(features);

  DebugSerial.println();
  DebugSerial.println("========================================");
  DebugSerial.println(title);
  DebugSerial.print("Classe prevista pelo ESP32-S3: ");
  DebugSerial.println(CLASS_NAMES[predictedClass]);
  printFeatureSummary(features);
  DebugSerial.println("========================================");
  DebugSerial.println("Comandos: n=sortear janela do dataset | l=inferencia live unica | r=leitura crua");
  DebugSerial.println("         m=liga/desliga monitor live do MPU6050");
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
  DebugSerial.println("Comandos: n=sortear janela do dataset | l=inferencia live unica | r=leitura crua");
  DebugSerial.println("         m=liga/desliga monitor live do MPU6050");
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
  DebugSerial.println("Comandos: n=sortear janela do dataset | l=inferencia live unica | r=leitura crua");
  DebugSerial.println("         m=liga/desliga monitor live do MPU6050");
}

void updateLiveMonitor() {
  unsigned long now = millis();
  if (!liveMonitorEnabled || now - lastLiveSampleMs < SAMPLE_INTERVAL_MS) {
    return;
  }

  lastLiveSampleMs = now;
  SensorReading reading = readMpu6050();
  if (!reading.ok) {
    return;
  }

  liveWindow[0][liveSampleIndex] = reading.accX;
  liveWindow[1][liveSampleIndex] = reading.accY;
  liveWindow[2][liveSampleIndex] = reading.accZ;
  liveWindow[3][liveSampleIndex] = reading.gyroX;
  liveWindow[4][liveSampleIndex] = reading.gyroY;
  liveWindow[5][liveSampleIndex] = reading.gyroZ;

  liveSampleIndex++;
  if (liveSampleIndex >= WINDOW_SIZE) {
    liveSampleIndex = 0;
    liveWindowReady = true;
  }

  if (liveWindowReady && now - lastLiveInferenceMs >= LIVE_INFERENCE_INTERVAL_MS) {
    lastLiveInferenceMs = now;
    runInferenceForWindow(liveWindow, "Monitor live do MPU6050 usando os sliders do Wokwi");
  }
}

void setup() {
  DebugSerial.begin(115200);
  delay(500);

  DebugSerial.println();
  DebugSerial.println("BOOT OK - Serial USB ativa em 115200");
  DebugSerial.println("========================================");
  DebugSerial.println("Projeto Final - IA Embarcada e Modelos Compactos");
  DebugSerial.println("Classificador de atividade humana com UCI HAR + ESP32-S3");
  DebugSerial.println("Modelo embarcado: MLP compacta com 1 camada oculta de 16 neuronios");
  DebugSerial.println("Modos disponiveis:");
  DebugSerial.println("  n = sortear uma janela real do dataset UCI HAR para comparacao");
  DebugSerial.println("  l = coletar 128 leituras do MPU6050 e inferir uma vez");
  DebugSerial.println("  r = mostrar uma leitura instantanea do MPU6050");
  DebugSerial.println("  m = ligar/desligar monitor live do MPU6050");
  DebugSerial.println("========================================");

  setupMpu6050();
  randomSeed((uint32_t)micros());
  DebugSerial.println("MPU6050 inicializado.");
  printMpu6050Once();
  DebugSerial.println();
  DebugSerial.println("Monitor live ligado.");
  DebugSerial.println("Mexa nos sliders do MPU6050 no Wokwi para alterar as leituras do sensor.");
  DebugSerial.println("A primeira previsao aparece depois que 128 leituras forem coletadas.");
}

void loop() {
  updateLiveMonitor();

  if (DebugSerial.available()) {
    char command = DebugSerial.read();
    if (command == 'n' || command == 'N') {
      chooseRandomDemoSample();
      runInferenceForCurrentSample();
    } else if (command == 'l' || command == 'L') {
      runLiveInferenceFromMpu6050();
    } else if (command == 'r' || command == 'R') {
      printMpu6050Once();
    } else if (command == 'm' || command == 'M') {
      liveMonitorEnabled = !liveMonitorEnabled;
      DebugSerial.println();
      DebugSerial.print("Monitor live do MPU6050: ");
      DebugSerial.println(liveMonitorEnabled ? "ligado" : "desligado");
    }
  }
}
