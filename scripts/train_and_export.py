from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATA_ROOT = Path("data/uci_har")
MODEL_DIR = Path("models")
FIRMWARE_DIR = Path("firmware/wokwi")

ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

SENSOR_FILES = [
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
]

FEATURE_NAMES = []
for sensor in SENSOR_FILES:
    FEATURE_NAMES.extend(
        [
            f"{sensor}_mean",
            f"{sensor}_std",
            f"{sensor}_min",
            f"{sensor}_max",
            f"{sensor}_rms",
        ]
    )


def find_dataset_base():
    matches = list(DATA_ROOT.rglob("UCI HAR Dataset"))
    for match in matches:
        if (match / "train" / "Inertial Signals").exists() and (
            match / "test" / "Inertial Signals"
        ).exists():
            return match
    raise FileNotFoundError(
        "Dataset nao encontrado. Rode primeiro: python scripts/download_dataset.py"
    )


def load_signal(base, split, sensor):
    path = base / split / "Inertial Signals" / f"{sensor}_{split}.txt"
    return np.loadtxt(path)


def extract_features(signals):
    features = []
    for signal in signals:
        features.append(signal.mean(axis=1))
        features.append(signal.std(axis=1))
        features.append(signal.min(axis=1))
        features.append(signal.max(axis=1))
        features.append(np.sqrt((signal * signal).mean(axis=1)))
    return np.column_stack(features)


def load_split(base, split):
    signals = [load_signal(base, split, sensor) for sensor in SENSOR_FILES]
    x = extract_features(signals)
    y = np.loadtxt(base / split / f"y_{split}.txt", dtype=np.int64) - 1
    return x, y, signals


def save_metrics(report_text, accuracy, compact_nodes, full_nodes, mlp_params, tree_accuracy):
    MODEL_DIR.mkdir(exist_ok=True)
    text = [
        "# Metricas do Modelo",
        "",
        "Modelo final embarcado: MLP compacta",
        f"Acuracia da MLP no conjunto de teste: {accuracy:.4f}",
        f"Parametros da MLP compacta: {mlp_params}",
        "",
        "Modelo anterior de comparacao: arvore de decisao compacta",
        f"Acuracia da arvore compacta: {tree_accuracy:.4f}",
        f"Nos da arvore compacta: {compact_nodes}",
        f"Nos da arvore completa de comparacao: {full_nodes}",
        "",
        "A compactacao final foi feita escolhendo uma MLP pequena, com apenas uma camada oculta de 16 neuronios.",
        "Isso melhora a metrica em relacao a arvore compacta e ainda permite embarcar o modelo como arrays C/C++.",
        "",
        "## Relatorio por classe",
        "",
        "```text",
        report_text.strip(),
        "```",
        "",
    ]
    (MODEL_DIR / "metrics.md").write_text("\n".join(text), encoding="utf-8")


def export_mlp_to_header(model):
    scaler = model.named_steps["standardscaler"]
    mlp = model.named_steps["mlpclassifier"]
    input_weights = mlp.coefs_[0]
    output_weights = mlp.coefs_[1]
    hidden_bias = mlp.intercepts_[0]
    output_bias = mlp.intercepts_[1]
    hidden_count = input_weights.shape[1]

    lines = [
        "#pragma once",
        "",
        "// Arquivo gerado por scripts/train_and_export.py.",
        "// Ele contem a MLP compacta treinada e pronta para embarcar.",
        "",
        f"const int FEATURE_COUNT = {len(FEATURE_NAMES)};",
        f"const int CLASS_COUNT = {len(ACTIVITY_NAMES)};",
        f"const int HIDDEN_COUNT = {hidden_count};",
        "",
        "const char* CLASS_NAMES[CLASS_COUNT] = {",
    ]
    lines.extend([f'  "{name}",' for name in ACTIVITY_NAMES])
    lines.append("};")
    lines.append("")
    lines.append("const char* FEATURE_NAMES[FEATURE_COUNT] = {")
    lines.extend([f'  "{name}",' for name in FEATURE_NAMES])
    lines.append("};")
    lines.append("")

    def array_1d(name, values):
        lines.append(f"const float {name}[{len(values)}] = {{")
        chunk = []
        for value in values:
            chunk.append(f"{float(value):.8f}f")
        for i in range(0, len(chunk), 8):
            lines.append("  " + ", ".join(chunk[i : i + 8]) + ",")
        lines.append("};")
        lines.append("")

    def array_2d(name, values, rows, cols):
        lines.append(f"const float {name}[{rows}][{cols}] = {{")
        for row in values:
            formatted = ", ".join(f"{float(value):.8f}f" for value in row)
            lines.append(f"  {{{formatted}}},")
        lines.append("};")
        lines.append("")

    array_1d("SCALER_MEAN", scaler.mean_)
    array_1d("SCALER_SCALE", scaler.scale_)
    array_2d("MLP_W1", input_weights, len(FEATURE_NAMES), hidden_count)
    array_1d("MLP_B1", hidden_bias)
    array_2d("MLP_W2", output_weights, hidden_count, len(ACTIVITY_NAMES))
    array_1d("MLP_B2", output_bias)

    FIRMWARE_DIR.mkdir(parents=True, exist_ok=True)
    (FIRMWARE_DIR / "model_data.h").write_text("\n".join(lines), encoding="utf-8")


def export_demo_windows(test_signals, y_test):
    samples = []
    used_classes = set()
    for index, label in enumerate(y_test):
        if int(label) not in used_classes:
            used_classes.add(int(label))
            sensor_windows = [signal[index].astype(float) for signal in test_signals]
            samples.append((int(label), sensor_windows))
        if len(samples) == len(ACTIVITY_NAMES):
            break

    lines = [
        "#pragma once",
        "",
        "// Janelas reais do dataset UCI HAR para demonstrar a inferencia no Wokwi.",
        "// Cada janela possui 128 leituras de 6 sinais: acelerometro x/y/z e giroscopio x/y/z.",
        "",
        "const int WINDOW_SIZE = 128;",
        f"const int DEMO_SAMPLE_COUNT = {len(samples)};",
        "",
        "const int DEMO_LABELS[DEMO_SAMPLE_COUNT] = {",
        "  " + ", ".join(str(label) for label, _ in samples) + ",",
        "};",
        "",
        "const float DEMO_WINDOWS[DEMO_SAMPLE_COUNT][6][WINDOW_SIZE] = {",
    ]

    for label, sensor_windows in samples:
        lines.append("  {")
        for window in sensor_windows:
            formatted = ", ".join(f"{value:.6f}f" for value in window)
            lines.append(f"    {{{formatted}}},")
        lines.append("  },")
    lines.append("};")

    FIRMWARE_DIR.mkdir(parents=True, exist_ok=True)
    (FIRMWARE_DIR / "demo_windows.h").write_text("\n".join(lines), encoding="utf-8")


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    FIRMWARE_DIR.mkdir(parents=True, exist_ok=True)

    base = find_dataset_base()
    x_train, y_train, _ = load_split(base, "train")
    x_test, y_test, test_signals = load_split(base, "test")

    full_model = DecisionTreeClassifier(random_state=42)
    full_model.fit(x_train, y_train)

    compact_tree = DecisionTreeClassifier(max_depth=6, min_samples_leaf=8, random_state=42)
    compact_tree.fit(x_train, y_train)

    mlp_model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(16,),
            activation="relu",
            solver="adam",
            alpha=0.0005,
            max_iter=800,
            early_stopping=True,
            n_iter_no_change=25,
            random_state=42,
            learning_rate_init=0.001,
        ),
    )
    mlp_model.fit(x_train, y_train)

    tree_predictions = compact_tree.predict(x_test)
    tree_accuracy = accuracy_score(y_test, tree_predictions)

    predictions = mlp_model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=ACTIVITY_NAMES)
    mlp = mlp_model.named_steps["mlpclassifier"]
    mlp_params = int(
        sum(weights.size for weights in mlp.coefs_)
        + sum(bias.size for bias in mlp.intercepts_)
    )

    save_metrics(
        report,
        accuracy,
        compact_tree.tree_.node_count,
        full_model.tree_.node_count,
        mlp_params,
        tree_accuracy,
    )
    export_mlp_to_header(mlp_model)
    export_demo_windows(test_signals, y_test)

    print(f"Acuracia MLP compacta: {accuracy:.4f}")
    print(f"Parametros da MLP compacta: {mlp_params}")
    print(f"Acuracia arvore compacta: {tree_accuracy:.4f}")
    print(f"Nos da arvore compacta: {compact_tree.tree_.node_count}")
    print(f"Nos da arvore completa: {full_model.tree_.node_count}")
    print("Arquivos gerados em models/ e firmware/wokwi/.")


if __name__ == "__main__":
    main()
