# cancer_recurrence_prediction/quantum_models.py

"""
Quantum model Variational Quantum Classifier (VQC) and a Quantum Kernel Classifier (QKC)

"""

import pennylane as qml
from pennylane import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve
from sklearn.svm import SVC
from tqdm import tqdm
import pickle

class VariationalClassifier:
    """
    A simple variational quantum classifier with training on noiseless simulator
    and evaluation under a noise toggle.
    """

    def __init__(self, n_qubits = 6, layers = 1, use_noise_in_test = False, noise_level = 0.05):
        self.n_qubits = n_qubits
        self.layers = layers
        self.noise_level = noise_level
        self.use_noise_in_test = use_noise_in_test

        # Initialize parameters: shape (2, n_qubits)
        self.params = np.random.randn(2 * layers, n_qubits, requires_grad = True)
        
        # Devices
        self.dev_train = qml.device("lightning.qubit", wires = n_qubits)
        self.dev_test = qml.device("default.mixed" if use_noise_in_test else "lightning.qubit", wires = n_qubits)

        # QNodes
        self.qnn_train = qml.QNode(self.circuit_train, self.dev_train, interface = "autograd")
        self.qnn_test = qml.QNode(self.circuit_test, self.dev_test, interface = "autograd")

        # Optimizer
        self.opt = qml.AdamOptimizer(stepsize = 0.05)

    def variational_circuit(self, params, x, noise = False):
        """
        Quantum circuit with angle embedding + variational ansatz.
        Each layer has:
            1. Parametrized R_Y rotations on all qubits
            2. CNOTs entangling every adjacent pair of qubits
            3. Another set of parametrized R_Y rotations on all qubits
        
        A simple noise model is applied through a Depolarizing Channel after each CNOT, that can be toggled in config.py
        """
        qml.AngleEmbedding(x, wires = range(self.n_qubits))

        for l in range(self.layers):
            for i in range(self.n_qubits):
                qml.RY(params[0][i], wires = i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires = [i, i + 1])
                if noise:
                    qml.DepolarizingChannel(self.noise_level, wires = i)
                    qml.DepolarizingChannel(self.noise_level, wires = i + 1)
            for i in range(self.n_qubits):
                qml.RY(params[1][i], wires = i)

    def circuit_train(self, params, x):
        """Noiseless circuit for training"""
        self.variational_circuit(params, x, noise = False)
        return qml.probs(wires = [0])

    def circuit_test(self, params, x):
        """Noisy/noiseless circuit for training"""
        self.variational_circuit(params, x, noise = self.use_noise_in_test)
        return qml.probs(wires = [0])

    def predict_probs(self, x, test = False):
        """Predict probabilities (between 0 and 1)"""
        if test:
            return self.qnn_test(self.params, x)[1]
        else:
            return self.qnn_train(self.params, x)[1]

    def cost(self, params, X, y):
        """Compute cost function"""
        predictions = [self.qnn_train(params, x)[1] for x in X]
        predictions = np.stack(predictions)
        predictions = np.clip(predictions, 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

    def fit(self, X, y, epochs = 30):
        """Train on noiseless simulator"""
        for epoch in range(epochs):
            self.params = self.opt.step(lambda w: self.cost(w, X, y), self.params)
            train_loss = self.cost(self.params, X, y)
            if epoch % 5 == 0:
                acc = self.score(X, y, test=False)
                print(f"Epoch {epoch:02d}: Loss = {train_loss:.4f}, Training Accuracy = {acc:.4f}")

    def predict(self, X, test = True):
        """Predict class labels (0 or 1)."""
        probs = [self.predict_probs(x, test=test) for x in X]
        return np.array(probs), (np.array(probs) > 0.5).astype(int)

    def score(self, X, y, test=True):
        """Return accuracy on data."""
        _, preds = self.predict(X, test=test)
        return np.mean(preds == y)

    def evaluate(self, X, y, test=True):
        """Return full evaluation metrics."""
        y_probs, y_preds = self.predict(X, test=test)
        metrics = {
            "accuracy": accuracy_score(y, y_preds),
            "precision": precision_score(y, y_preds),
            "recall": recall_score(y, y_preds),
            "f1_score": f1_score(y, y_preds),
            "roc_curve": roc_curve(y, y_probs),
            "pr_curve": precision_recall_curve(y, y_probs)
        }
        return metrics
    
    def save(self, filepath):
        """Save parameters of model"""
        data = {
            "params": self.params,
            "n_qubits": self.n_qubits,
            "layers": self.layers,
            "noise_level": self.noise_level,
            "use_noise_in_test": self.use_noise_in_test
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath):
        """Create model from loaded parameters"""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        model = cls(
            n_qubits=data["n_qubits"],
            layers = data["layers"],
            noise_level = data["noise_level"],
            use_noise_in_test = data["use_noise_in_test"]
        )
        model.params = data["params"]
        return model

class QuantumKernelClassifier:
    
    """
    A kernel-based quantum classifier using a fidelity kernel.
    - Uses AngleEmbedding
    - Kernel matrix K(x, y) = |⟨ϕ(x)|ϕ(y)⟩|²
    - Trained using classical SVM with precomputed kernel matrix
    
    A simple noise model is applied through a Depolarizing Channel on each qubit, that can be toggled in config.py 
    """

    def __init__(self, n_qubits = 6, use_noise_in_test = False, noise_level = 0.1):
        self.n_qubits = n_qubits
        self.noise_level = noise_level
        self.use_noise_in_test = use_noise_in_test

        # Quantum devices
        self.dev_train = qml.device("lightning.qubit", wires = n_qubits)
        self.dev_test = qml.device("default.mixed" if use_noise_in_test else "lightning.qubit", wires = n_qubits)

        # QNodes
        self.kernel_train_qnode = qml.QNode(self.kernel_circuit_train, self.dev_train)
        self.kernel_test_qnode = qml.QNode(self.kernel_circuit_test, self.dev_test)

        # Model placeholder
        self.svc = None
        self.X_train_ref = None

    def encoding(self, x, noise = False):
        """Encoding circuit with angle embedding and noise channel"""
        qml.AngleEmbedding(x, wires = range(self.n_qubits))
        if noise:
            for i in range(self.n_qubits):
                qml.DepolarizingChannel(self.noise_level, wires = i)

    def kernel_circuit_train(self, x1, x2):
        """Kernel circuit (noiseless) for training"""
        self.encoding(x1, noise = False)
        qml.adjoint(self.encoding)(x2, noise = False)
        return qml.probs(wires = range(self.n_qubits))

    def kernel_circuit_test(self, x1, x2):
        """Kernel circuit (noisy/noiseless) for training"""
        self.encoding(x1, noise = self.use_noise_in_test)
        qml.adjoint(self.encoding)(x2, noise = False)
        return qml.probs(wires = range(self.n_qubits))

    def kernel_train(self, a, b):
        """Apply projector |00...0><00...0| on training circuit"""
        return self.kernel_train_qnode(a, b)[0]

    def kernel_test(self, a, b):
        """Apply projector |00...0><00...0| on test circuit"""
        return self.kernel_test_qnode(a, b)[0]

    def kernel_matrix(self, A, B, kernel_func):
        """Compute Gram matrix between A and B using kernel_func."""
        return np.array([[kernel_func(a, b) for b in B] for a in tqdm(A, desc = "Kernel matrix")])

    def fit(self, X_train, y_train):
        """Compute train kernel matrix and fit SVM."""
        self.X_train_ref = X_train
        K_train = self.kernel_matrix(X_train, X_train, self.kernel_train)
        self.svc = SVC(kernel = 'precomputed', probability = True)
        self.svc.fit(K_train, y_train)

    def predict(self, X_test):
        """Predict class labels on test data using precomputed kernel."""
        if self.X_train_ref is None or self.svc is None:
            raise RuntimeError("Call fit() before predict().")
        K_test = self.kernel_matrix(X_test, self.X_train_ref, self.kernel_test)
        return self.svc.predict(K_test)

    def predict_probs(self, X_test):
        """Return predicted probabilities (class 1)."""
        if self.X_train_ref is None or self.svc is None:
            raise RuntimeError("Call fit() before predict_probs().")
        K_test = self.kernel_matrix(X_test, self.X_train_ref, self.kernel_test)
        return self.svc.predict_proba(K_test)[:, 1]

    def evaluate(self, X_test, y_test):
        """Return evaluation metrics on test set."""
        y_probs = self.predict_probs(X_test)
        y_preds = (y_probs > 0.5).astype(int)

        return {
            "accuracy": accuracy_score(y_test, y_preds),
            "precision": precision_score(y_test, y_preds),
            "recall": recall_score(y_test, y_preds),
            "f1_score": f1_score(y_test, y_preds),
            "roc_curve": roc_curve(y_test, y_probs),
            "pr_curve": precision_recall_curve(y_test, y_probs)
        }
    
    def save(self, filepath):
        """Save parameters of model"""
        data = {
            "svc": self.svc,
            "X_train_ref": self.X_train_ref,
            "n_qubits": self.n_qubits,
            "noise_level": self.noise_level,
            "use_noise_in_test": self.use_noise_in_test
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath):
        """Create model from loaded parameters"""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        model = cls(
            n_qubits = data["n_qubits"],
            noise_level = data["noise_level"],
            use_noise_in_test = data["use_noise_in_test"]
        )
        model.svc = data["svc"]
        model.X_train_ref = data["X_train_ref"]
        return model