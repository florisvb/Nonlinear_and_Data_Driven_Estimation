import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

import json
import h5py
import numpy as np
import os

# ============================================================================
# FAST INFERENCE OPTIMIZATION
# ============================================================================

def create_fast_inference_model(trained_model):
    """
    Create an optimized version of the model for fast inference using XLA compilation.
    Removes training-specific components (dropout, regularization).
    
    Note: The first prediction will be slower as it includes compilation time.
    Subsequent predictions will be fast.
    
    Parameters:
    -----------
    trained_model : JacobianRegularizedModel
        The trained model
    
    Returns:
    --------
    inference_func : tf.function
        Optimized inference function for predictions
    
    Example:
    --------
    >>> fast_predict = create_fast_inference_model(model)
    >>> # First call is slow (compilation)
    >>> predictions = fast_predict(X_core_batch, X_aux_batch)
    >>> # Subsequent calls are fast
    >>> predictions = fast_predict(X_core_batch2, X_aux_batch2)
    """
    # Get the base functional model (without custom training logic)
    inputs = trained_model.inputs
    outputs = trained_model.outputs
    
    # Create inference-only model (dropout disabled)
    inference_model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Get only trainable weights (excludes metric trackers)
    trainable_weights = trained_model.trainable_weights
    inference_trainable_weights = inference_model.trainable_weights
    
    # Copy weights
    for target_weight, source_weight in zip(inference_trainable_weights, trainable_weights):
        target_weight.assign(source_weight)
    
    # Create optimized prediction function
    @tf.function(jit_compile=True)  # XLA compilation for speed
    def fast_predict(x_core, x_aux):
        return inference_model([x_core, x_aux], training=False)
    
    print("✓ Fast inference model created with XLA compilation")
    print("  First prediction will be slow (includes compilation)")
    print("  Subsequent predictions will be fast")
    print("  Use: predictions = fast_predict(X_core_tensor, X_aux_tensor)")
    print("  Note: Inputs must be TensorFlow tensors (use tf.constant())")
    
    return fast_predict


def create_inference_only_model(trained_model):
    """
    Create a simplified inference-only model without custom training components.
    This is faster than using the JacobianRegularizedModel for predictions.
    
    Parameters:
    -----------
    trained_model : JacobianRegularizedModel
        The trained model
    
    Returns:
    --------
    inference_model : keras.Model
        Standard Keras model (no custom training step) for fast inference
    
    Example:
    --------
    >>> inference_model = create_inference_only_model(model)
    >>> predictions = inference_model.predict([X_core, X_aux], batch_size=1024)
    """
    # Extract the functional model structure
    inputs = trained_model.inputs
    outputs = trained_model.outputs
    
    # Create standard Keras model
    inference_model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Get only the trainable weights (excludes metric trackers)
    trainable_weights = trained_model.trainable_weights
    inference_trainable_weights = inference_model.trainable_weights
    
    # Verify weight shapes match
    if len(trainable_weights) != len(inference_trainable_weights):
        raise ValueError(
            f"Weight mismatch: trained model has {len(trainable_weights)} "
            f"trainable weights but inference model has {len(inference_trainable_weights)}"
        )
    
    # Copy weights layer by layer
    for target_weight, source_weight in zip(inference_trainable_weights, trainable_weights):
        target_weight.assign(source_weight)
    
    print("✓ Inference-only model created")
    print("  This is a standard Keras model without custom training logic")
    print("  Dropout is automatically disabled during inference")
    
    return inference_model


def benchmark_inference_speed(model, X_core, X_aux, num_iterations=100):
    """
    Benchmark inference speed of a model.
    
    Parameters:
    -----------
    model : keras.Model or callable
        Model or prediction function to benchmark
    X_core : np.ndarray
        Core features
    X_aux : np.ndarray
        Auxiliary features
    num_iterations : int
        Number of iterations for timing
    
    Returns:
    --------
    dict : Timing statistics
    """
    import time
    
    # Warmup
    if callable(model) and not isinstance(model, keras.Model):
        # TensorFlow function
        _ = model(tf.constant(X_core[:1], dtype=tf.float32),
                  tf.constant(X_aux[:1], dtype=tf.float32))
    else:
        # Keras model
        _ = model.predict([X_core[:1], X_aux[:1]], verbose=0)
    
    # Time multiple iterations
    times = []
    for _ in range(num_iterations):
        start = time.time()
        
        if callable(model) and not isinstance(model, keras.Model):
            _ = model(tf.constant(X_core, dtype=tf.float32),
                     tf.constant(X_aux, dtype=tf.float32))
        else:
            _ = model.predict([X_core, X_aux], verbose=0)
        
        times.append(time.time() - start)
    
    times = np.array(times)
    samples_per_sec = len(X_core) / np.mean(times)
    
    results = {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'samples_per_second': samples_per_sec,
        'batch_size': len(X_core)
    }
    
    print(f"Inference Benchmark Results ({num_iterations} iterations):")
    print(f"  Mean time: {results['mean_time']*1000:.2f} ms")
    print(f"  Std time: {results['std_time']*1000:.2f} ms")
    print(f"  Throughput: {results['samples_per_second']:.0f} samples/sec")
    
    return results


# ============================================================================
# MODEL SAVING AND LOADING FUNCTIONS
# ============================================================================

def save_model_complete(model, filepath, core_architecture, aux_architecture, 
                       combined_architecture, input_architecture, 
                       training_parameters):
    """
    Save model weights and complete configuration for easy reloading.
    """
    
    # Save weights manually layer-by-layer for maximum compatibility
    h5_path = f"{filepath}.weights.h5"
    
    print("Saving weights layer-by-layer for compatibility...")
    with h5py.File(h5_path, 'w') as f:
        for layer in model.layers:
            if len(layer.weights) > 0:
                layer_group = f.create_group(layer.name)
                for weight in layer.weights:
                    weight_name = weight.name.split('/')[-1].split(':')[0]
                    layer_group.create_dataset(weight_name, data=weight.numpy())
                print(f"  Saved layer: {layer.name}")
    
    print(f"✓ Saved weights to: {h5_path}")
    
    # Save complete configuration
    config = {
        'core_architecture': core_architecture,
        'aux_architecture': aux_architecture,
        'combined_architecture': combined_architecture,
        'input_architecture': input_architecture,
        'training_parameters': training_parameters
    }
    
    config_path = f"{filepath}.config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Saved configuration to: {config_path}")
    
    print(f"\nModel saved successfully!")
    print(f"To load: model, dropout_layer, config = load_model_complete('{filepath}')")

def load_model_complete(filepath, compile_model=True, learning_rate=0.001):
    """
    Load model weights and configuration.
    """
    
    # Load configuration
    config_path = f"{filepath}.config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"✓ Loaded configuration from: {config_path}")
    
    # Rebuild model with explicit layer names
    model, dropout_layer = build_auxiliary_dropout_model(
        core_input_dim=config['input_architecture']['core_input_dim'],
        aux_input_dim=config['input_architecture']['aux_input_dim'],
        core_architecture=config['core_architecture'],
        aux_architecture=config['aux_architecture'],
        combined_architecture=config['combined_architecture'],
        jacobian_weight=config['training_parameters']['jacobian_weight'],
        sv_weight=config['training_parameters']['sv_weight']
    )
    
    # Compile if requested
    if compile_model:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        print(f"✓ Model compiled with learning_rate={learning_rate}")
    
    # Load weights from .h5 file
    h5_path = f"{filepath}.weights.h5"
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Weights file not found: {h5_path}")
    
    print(f"Loading weights from: {h5_path}")
    
    # Load layer-by-layer (matches how we saved)
    with h5py.File(h5_path, 'r') as f:
        layers_loaded = 0
        
        for layer in model.layers:
            if len(layer.weights) > 0:
                if layer.name not in f:
                    print(f"  ⚠ Warning: Layer '{layer.name}' not found in file")
                    continue
                
                layer_group = f[layer.name]
                weight_values = []
                
                for weight in layer.weights:
                    weight_name = weight.name.split('/')[-1].split(':')[0]
                    if weight_name in layer_group:
                        weight_values.append(np.array(layer_group[weight_name]))
                    else:
                        print(f"  ⚠ Warning: Weight '{weight_name}' not found in layer '{layer.name}'")
                        break
                
                if len(weight_values) == len(layer.weights):
                    layer.set_weights(weight_values)
                    layers_loaded += 1
                    print(f"  ✓ Loaded: {layer.name}")
        
        print(f"\n✓ Loaded {layers_loaded} layers from: {h5_path}")
    
    print(f"\n✓ Model loaded successfully!")
    print(f"  Core input dim: {config['input_architecture']['core_input_dim']}")
    print(f"  Aux input dim: {config['input_architecture']['aux_input_dim']}")
    
    return model, dropout_layer, config



# ============================================================================
# DATA PREPARATION HELPER FUNCTION
# ============================================================================

def prepare_train_test_split(data, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    
    Parameters:
    -----------
    data : dict
        Dictionary with keys 'X_core_train', 'X_aux_train', 'y_train'
        Each value should be a numpy array
    test_size : float
        Fraction of data to use for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    train_test_data : dict
        Dictionary with keys:
        - 'X_core_train': Core training features
        - 'X_core_test': Core test features
        - 'X_aux_train': Auxiliary training features
        - 'X_aux_test': Auxiliary test features
        - 'y_train': Training targets
        - 'y_test': Test targets
    
    Example:
    --------
    >>> data = {
    ...     'X_core_train': core_data,
    ...     'X_aux_train': aux_data,
    ...     'y_train': output_data
    ... }
    >>> split_data = prepare_train_test_split(data, test_size=0.2)
    >>> # Now use split_data['X_core_train'], split_data['X_core_test'], etc.
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Extract data
    X_core = data['X_core_train']
    X_aux = data['X_aux_train']
    y = data['y_train']
    
    # Check that all arrays have the same number of samples
    n_samples = len(X_core)
    assert len(X_aux) == n_samples, "X_core and X_aux must have same number of samples"
    assert len(y) == n_samples, "y must have same number of samples as X_core and X_aux"
    
    # Create random indices for splitting
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Calculate split point
    split_idx = int(n_samples * (1 - test_size))
    
    # Split indices
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # Split data
    train_test_data = {
        'X_core_train': X_core[train_indices],
        'X_core_test': X_core[test_indices],
        'X_aux_train': X_aux[train_indices],
        'X_aux_test': X_aux[test_indices],
        'y_train': y[train_indices],
        'y_test': y[test_indices]
    }
    
    print(f"Data split complete:")
    print(f"  Training samples: {len(train_indices)} ({(1-test_size)*100:.0f}%)")
    print(f"  Test samples: {len(test_indices)} ({test_size*100:.0f}%)")
    print(f"  Core features: {X_core.shape[1]}")
    print(f"  Aux features: {X_aux.shape[1]}")
    if len(y.shape) == 1:
        print(f"  Output dimension: 1")
    else:
        print(f"  Output dimension: {y.shape[1]}")
    
    return train_test_data


# ============================================================================
# CUSTOM MODEL AND CALLBACKS
# ============================================================================

class JacobianRegularizedModel(keras.Model):
    """Model with dual Jacobian regularization: smooth gradients + large singular values"""
    
    def __init__(self, *args, jacobian_weight=0.01, sv_weight=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.jacobian_weight = jacobian_weight  # Smoothness penalty
        self.sv_weight = sv_weight  # Singular value penalty
        self.jacobian_loss_tracker = keras.metrics.Mean(name="jacobian_smoothness")
        self.sv_loss_tracker = keras.metrics.Mean(name="sv_loss")
    
    def get_config(self):
        """Enable serialization of the model"""
        config = super().get_config()
        config.update({
            'jacobian_weight': self.jacobian_weight,
            'sv_weight': self.sv_weight,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Enable deserialization of the model"""
        return cls(**config)
    
    def train_step(self, data):
        x, y = data
        
        # Handle both single input and list/tuple of inputs
        if isinstance(x, (list, tuple)):
            x_core = x[0]
            x_aux = x[1]
        else:
            # If single input, split based on model's input shapes
            core_dim = self.input[0].shape[-1]
            x_core = x[:, :core_dim]
            x_aux = x[:, core_dim:]
        
        with tf.GradientTape() as tape:
            # Forward pass
            if isinstance(x, (list, tuple)):
                y_pred = self(x, training=True)
            else:
                y_pred = self([x_core, x_aux], training=True)
            
            # Base loss (MSE)
            base_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            
            # Compute Jacobian for current batch
            with tf.GradientTape() as jacobian_tape:
                # Convert to tensor and watch
                x_core_tensor = tf.convert_to_tensor(x_core, dtype=tf.float32)
                jacobian_tape.watch(x_core_tensor)
                
                if isinstance(x, (list, tuple)):
                    y_pred_for_jacobian = self([x_core_tensor, x_aux], training=True)
                else:
                    y_pred_for_jacobian = self([x_core_tensor, x_aux], training=True)
            
            jacobian = jacobian_tape.gradient(y_pred_for_jacobian, x_core_tensor)
            
            # Penalty 1: Encourage large singular values (Jacobian norms >= 1.0)
            jacobian_norms = tf.norm(jacobian, axis=1, keepdims=True)
            target_sv = 1.0
            sv_loss = tf.reduce_mean(tf.square(tf.maximum(0.0, target_sv - jacobian_norms)))
            
            # Penalty 2: Encourage smooth Jacobian (Hessian regularization)
            epsilon = 0.01
            x_core_perturbed = x_core_tensor + tf.random.normal(tf.shape(x_core_tensor)) * epsilon
            
            with tf.GradientTape() as jacobian_tape2:
                jacobian_tape2.watch(x_core_perturbed)
                y_pred_perturbed = self([x_core_perturbed, x_aux], training=True)
            
            jacobian_perturbed = jacobian_tape2.gradient(y_pred_perturbed, x_core_perturbed)
            jacobian_smoothness = tf.reduce_mean(tf.square(jacobian - jacobian_perturbed))
            
            # Total loss
            total_loss = base_loss + self.sv_weight * sv_loss + self.jacobian_weight * jacobian_smoothness
        
        # Update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        self.jacobian_loss_tracker.update_state(jacobian_smoothness)
        self.sv_loss_tracker.update_state(sv_loss)
        
        # Return metrics
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['jacobian_smoothness'] = self.jacobian_loss_tracker.result()
        metrics['sv_loss'] = self.sv_loss_tracker.result()
        return metrics
    
    @property
    def metrics(self):
        return super().metrics + [self.jacobian_loss_tracker, self.sv_loss_tracker]


class CurriculumDropoutCallback(keras.callbacks.Callback):
    """Callback for curriculum learning: gradually increase dropout rate"""
    
    def __init__(self, dropout_layer, val_data_with_aux, val_data_without_aux, 
                 val_labels, initial_rate=0.1, final_rate=0.9, warmup_epochs=30):
        super().__init__()
        self.dropout_layer = dropout_layer
        self.val_data_with_aux = val_data_with_aux
        self.val_data_without_aux = val_data_without_aux
        self.val_labels = val_labels
        self.initial_rate = initial_rate
        self.final_rate = final_rate
        self.warmup_epochs = warmup_epochs
        self.dropout_history = []
        self.mae_with_aux_history = []
        self.mae_without_aux_history = []
    
    def on_epoch_begin(self, epoch, logs=None):
        # Gradually increase dropout rate
        if epoch < self.warmup_epochs:
            progress = epoch / self.warmup_epochs
            current_rate = (self.initial_rate + 
                          (self.final_rate - self.initial_rate) * progress)
        else:
            current_rate = self.final_rate
        
        self.dropout_layer.rate = current_rate
        self.dropout_history.append(current_rate)
    
    def on_epoch_end(self, epoch, logs=None):
        logs['dropout_rate'] = self.dropout_layer.rate
        
        # Evaluate with auxiliary features
        preds_with_aux = self.model.predict(self.val_data_with_aux, verbose=0)
        
        # Handle both single and multi-output cases
        if len(preds_with_aux.shape) == 1:
            preds_with_aux = preds_with_aux.reshape(-1, 1)
        if len(self.val_labels.shape) == 1:
            val_labels_reshaped = self.val_labels.reshape(-1, 1)
        else:
            val_labels_reshaped = self.val_labels
            
        mae_with_aux = np.mean(np.abs(val_labels_reshaped - preds_with_aux))
        self.mae_with_aux_history.append(mae_with_aux)
        
        # Evaluate without auxiliary features (zeros)
        preds_without_aux = self.model.predict(self.val_data_without_aux, verbose=0)
        
        # Handle both single and multi-output cases
        if len(preds_without_aux.shape) == 1:
            preds_without_aux = preds_without_aux.reshape(-1, 1)
            
        mae_without_aux = np.mean(np.abs(val_labels_reshaped - preds_without_aux))
        self.mae_without_aux_history.append(mae_without_aux)
        
        logs['val_mae_with_aux'] = mae_with_aux
        logs['val_mae_without_aux'] = mae_without_aux


# ============================================================================
# MODEL BUILDING FUNCTION
# ============================================================================

def build_auxiliary_dropout_model(
    core_input_dim,
    aux_input_dim,
    core_architecture,
    aux_architecture,
    combined_architecture,
    dropout_layer_name='aux_dropout',
    initial_dropout=0.1,
    jacobian_weight=0.01,
    sv_weight=0.01
):
    """
    Build a model with separate core and auxiliary feature branches.
    
    Parameters:
    -----------
    core_input_dim : int
        Dimension of core features (always available)
    aux_input_dim : int
        Dimension of auxiliary features (may not be available at inference)
    core_architecture : list of dict
        List of layer specs for core branch, e.g.:
        [{'units': 16, 'activation': 'relu'}, {'units': 8, 'activation': 'relu'}]
    aux_architecture : list of dict
        List of layer specs for auxiliary branch (after dropout)
    combined_architecture : list of dict
        List of layer specs for combined features, including output layer
    dropout_layer_name : str
        Name for the dropout layer (used to access it for curriculum learning)
    initial_dropout : float
        Initial dropout rate (will be adjusted by curriculum callback)
    jacobian_weight : float
        Weight for Jacobian smoothness regularization
    sv_weight : float
        Weight for singular value regularization
    
    Returns:
    --------
    model : JacobianRegularizedModel
        Compiled Keras model
    dropout_layer : keras.layers.Dropout
        Reference to dropout layer for curriculum learning
    """
    # Input layers with explicit names
    core_input = keras.layers.Input(shape=(core_input_dim,), name='core_features')
    aux_input = keras.layers.Input(shape=(aux_input_dim,), name='aux_features')
    
    # Core branch (no dropout) with explicit layer names
    core_branch = core_input
    for i, layer_spec in enumerate(core_architecture):
        # Add explicit name to layer_spec
        layer_spec_with_name = layer_spec.copy()
        layer_spec_with_name['name'] = f'core_dense_{i}'
        core_branch = keras.layers.Dense(**layer_spec_with_name)(core_branch)
    
    # Auxiliary branch (with dropout) with explicit layer names
    dropout_layer = keras.layers.Dropout(initial_dropout, name=dropout_layer_name)
    aux_branch = dropout_layer(aux_input)
    for i, layer_spec in enumerate(aux_architecture):
        # Add explicit name to layer_spec
        layer_spec_with_name = layer_spec.copy()
        layer_spec_with_name['name'] = f'aux_dense_{i}'
        aux_branch = keras.layers.Dense(**layer_spec_with_name)(aux_branch)
    
    # Combine branches with explicit name
    combined = keras.layers.concatenate([core_branch, aux_branch], name='concatenate')
    for i, layer_spec in enumerate(combined_architecture):
        # Add explicit name to layer_spec
        layer_spec_with_name = layer_spec.copy()
        layer_spec_with_name['name'] = f'combined_dense_{i}'
        combined = keras.layers.Dense(**layer_spec_with_name)(combined)
    
    # Create model with Jacobian regularization
    model = JacobianRegularizedModel(
        inputs=[core_input, aux_input],
        outputs=combined,
        jacobian_weight=jacobian_weight,
        sv_weight=sv_weight
    )
    
    return model, dropout_layer


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_with_curriculum(
    model,
    dropout_layer,
    X_core_train,
    X_aux_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    initial_dropout=0.1,
    final_dropout=0.9,
    warmup_fraction=0.8,
    verbose=1
):
    """
    Train model with curriculum learning on dropout rate.
    
    Parameters:
    -----------
    model : keras.Model
        Model to train
    dropout_layer : keras.layers.Dropout
        Dropout layer to adjust during training
    X_core_train : np.ndarray
        Core training features
    X_aux_train : np.ndarray
        Auxiliary training features
    y_train : np.ndarray
        Training targets
    validation_split : float
        Fraction of data to use for validation
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    initial_dropout : float
        Starting dropout rate
    final_dropout : float
        Final dropout rate
    warmup_fraction : float
        Fraction of total epochs to use for ramping up dropout (default: 0.8)
        For example, 0.8 means dropout increases for first 80% of epochs,
        then stays constant for the last 20%
    verbose : int
        Verbosity level (0, 1, or 2)
    
    Returns:
    --------
    history : keras.callbacks.History
        Training history
    curriculum_callback : CurriculumDropoutCallback
        Callback with dropout and performance history
    X_core_val, X_aux_val, y_val : np.ndarray
        Validation data (for later evaluation)
    """
    # Calculate warmup epochs based on total epochs
    warmup_epochs = int(epochs * warmup_fraction)
    
    print(f"Curriculum learning schedule:")
    print(f"  Total epochs: {epochs}")
    print(f"  Warmup epochs: {warmup_epochs} ({warmup_fraction*100:.0f}% of training)")
    print(f"  Dropout: {initial_dropout:.2f} → {final_dropout:.2f}")
    print(f"  Final dropout reached at epoch {warmup_epochs}")
    
    # Split into train and validation
    val_size = int(validation_split * len(X_core_train))
    train_size = len(X_core_train) - val_size
    
    X_core_train_split = X_core_train[:train_size]
    X_aux_train_split = X_aux_train[:train_size]
    y_train_split = y_train[:train_size]
    
    X_core_val = X_core_train[train_size:]
    X_aux_val = X_aux_train[train_size:]
    y_val = y_train[train_size:]
    
    # Prepare validation data for callback
    val_data_with_aux = [X_core_val, X_aux_val]
    val_data_without_aux = [X_core_val, np.zeros_like(X_aux_val)]
    
    # Create curriculum callback
    curriculum_callback = CurriculumDropoutCallback(
        dropout_layer=dropout_layer,
        val_data_with_aux=val_data_with_aux,
        val_data_without_aux=val_data_without_aux,
        val_labels=y_val,
        initial_rate=initial_dropout,
        final_rate=final_dropout,
        warmup_epochs=warmup_epochs
    )
    
    # Train
    history = model.fit(
        [X_core_train_split, X_aux_train_split],
        y_train_split,
        validation_data=([X_core_val, X_aux_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[curriculum_callback],
        verbose=verbose
    )
    
    return history, curriculum_callback, X_core_val, X_aux_val, y_val


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model, X_core, X_aux, y_true, dataset_name="Test"):
    """
    Evaluate model with and without auxiliary features.
    
    Parameters:
    -----------
    model : keras.Model
        Trained model
    X_core : np.ndarray
        Core features
    X_aux : np.ndarray
        Auxiliary features
    y_true : np.ndarray
        True targets
    dataset_name : str
        Name of dataset for printing
    
    Returns:
    --------
    results : dict
        Dictionary with predictions and metrics
    """
    # Predictions without auxiliary features
    preds_without_aux = model.predict([X_core, np.zeros_like(X_aux)], verbose=0)
    
    # Predictions with auxiliary features
    preds_with_aux = model.predict([X_core, X_aux], verbose=0)
    
    # Handle both single and multi-output cases
    if len(y_true.shape) == 1:
        y_true_reshaped = y_true.reshape(-1, 1)
    else:
        y_true_reshaped = y_true
        
    if len(preds_without_aux.shape) == 1:
        preds_without_aux = preds_without_aux.reshape(-1, 1)
    if len(preds_with_aux.shape) == 1:
        preds_with_aux = preds_with_aux.reshape(-1, 1)
    
    mae_without_aux = np.mean(np.abs(y_true_reshaped - preds_without_aux))
    mae_with_aux = np.mean(np.abs(y_true_reshaped - preds_with_aux))
    
    # Print results
    print(f"\n{'='*60}")
    print(f"{dataset_name.upper()} SET PERFORMANCE:")
    print(f"{'='*60}")
    print(f"MAE without auxiliary features: {mae_without_aux:.4f}")
    print(f"MAE with auxiliary features:    {mae_with_aux:.4f}")
    
    if mae_with_aux < mae_without_aux:
        improvement = ((mae_without_aux - mae_with_aux) / mae_without_aux * 100)
        print(f"✓ Aux features improve performance by {improvement:.2f}%")
    else:
        diff = ((mae_with_aux - mae_without_aux) / mae_with_aux * 100)
        print(f"⚠ Without aux performs {diff:.2f}% better")
    
    return {
        'predictions_without_aux': preds_without_aux,
        'predictions_with_aux': preds_with_aux,
        'mae_without_aux': mae_without_aux,
        'mae_with_aux': mae_with_aux
    }


# ============================================================================
# VISUALIZATION FUNCTION
# ============================================================================

def visualize_results(
    history,
    curriculum_callback,
    val_results,
    test_results,
    y_val,
    y_test,
    model,
    X_core_test,
    X_aux_test,
    fontsize=8,
):
    """
    Create comprehensive visualization of training and results.
    
    Parameters:
    -----------
    history : keras.callbacks.History
        Training history
    curriculum_callback : CurriculumDropoutCallback
        Callback with dropout history
    val_results : dict
        Validation results from evaluate_model
    test_results : dict
        Test results from evaluate_model
    y_val : np.ndarray
        Validation targets
    y_test : np.ndarray
        Test targets
    model : keras.Model
        Trained model
    X_core_test : np.ndarray
        Test core features (for Jacobian analysis)
    X_aux_test : np.ndarray
        Test aux features (for Jacobian analysis)
    """
    # Compute Jacobian for test set
    with tf.GradientTape() as tape:
        X_core_test_tf = tf.constant(X_core_test, dtype=tf.float32)
        X_aux_test_tf = tf.constant(np.zeros_like(X_aux_test), dtype=tf.float32)
        tape.watch(X_core_test_tf)
        predictions_tf = model([X_core_test_tf, X_aux_test_tf], training=False)
    
    jacobian_test = tape.gradient(predictions_tf, X_core_test_tf).numpy()
    jacobian_norms = np.linalg.norm(jacobian_test, axis=1)
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.4)
    
    # Row 1: Training dynamics
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Row 2: Validation set
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Row 3: Test set
    ax7 = fig.add_subplot(gs[2, 0])
    ax8 = fig.add_subplot(gs[2, 1])
    ax9 = fig.add_subplot(gs[2, 2])
    
    # Row 4: Jacobian analysis
    ax10 = fig.add_subplot(gs[3, 0])
    ax11 = fig.add_subplot(gs[3, 1])
    ax12 = fig.add_subplot(gs[3, 2])
    
    # Dropout schedule
    ax1.plot(curriculum_callback.dropout_history, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Dropout Rate')
    ax1.set_title('Curriculum Learning: Dropout Schedule')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=curriculum_callback.final_rate, color='r', linestyle='--', 
                alpha=0.5, label='Target rate')
    ax1.legend(fontsize=int(fontsize*2/3))
    
    # Performance over time
    ax2.plot(curriculum_callback.mae_with_aux_history, 
             label='With Aux Features', linewidth=2, color='skyblue')
    ax2.plot(curriculum_callback.mae_without_aux_history, 
             label='Without Aux Features', linewidth=2, color='coral')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation MAE')
    ax2.set_title('Performance Over Time')
    ax2.legend(fontsize=int(fontsize*2/3))
    ax2.grid(True, alpha=0.3)
    
    # Performance gap
    gap = (np.array(curriculum_callback.mae_without_aux_history) - 
           np.array(curriculum_callback.mae_with_aux_history))
    ax3.plot(gap, linewidth=2, color='green')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MAE Gap')
    ax3.set_title('Performance Gap\n(Without Aux - With Aux)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Validation predictions
    plot_predictions(ax4, y_val, val_results['predictions_without_aux'], 
                     val_results['mae_without_aux'], 'Validation: Without Aux', 'coral')
    plot_predictions(ax5, y_val, val_results['predictions_with_aux'], 
                     val_results['mae_with_aux'], 'Validation: With Aux', 'skyblue')
    plot_comparison_bars(ax6, val_results['mae_without_aux'], 
                         val_results['mae_with_aux'], 'Validation Performance')
    
    # Test predictions
    plot_predictions(ax7, y_test, test_results['predictions_without_aux'], 
                     test_results['mae_without_aux'], 'Test: Without Aux', 'coral')
    plot_predictions(ax8, y_test, test_results['predictions_with_aux'], 
                     test_results['mae_with_aux'], 'Test: With Aux', 'skyblue')
    plot_comparison_bars(ax9, test_results['mae_without_aux'], 
                         test_results['mae_with_aux'], 'Test Performance')
    
    # Jacobian analysis
    if 'jacobian_smoothness' in history.history:
        ax10.plot(history.history['jacobian_smoothness'], linewidth=2, 
                  color='purple', label='Smoothness')
        ax10_twin = ax10.twinx()
        ax10_twin.plot(history.history['sv_loss'], linewidth=2, 
                       color='orange', label='SV Loss')
        ax10.set_xlabel('Epoch')
        ax10.set_ylabel('Jacobian Smoothness Loss', color='purple')
        ax10_twin.set_ylabel('Singular Value Loss', color='orange')
        ax10.set_title('Regularization Losses Over Time')
        ax10.grid(True, alpha=0.3)
        ax10.legend(loc='upper left', fontsize=int(fontsize*2/3))
        ax10_twin.legend(loc='upper right', fontsize=int(fontsize*2/3))
    
    # Singular value distribution
    ax11.hist(jacobian_norms, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax11.axvline(jacobian_norms.mean(), color='red', linestyle='--', 
                 linewidth=2, label='Mean')
    ax11.axvline(1.0, color='green', linestyle='--', linewidth=2, label='Target (1.0)')
    ax11.set_xlabel('Jacobian Norm (Singular Value)')
    ax11.set_ylabel('Frequency')
    ax11.set_title(f'Singular Value Distribution Mean: {jacobian_norms.mean():.4f}')
    ax11.legend(fontsize=int(fontsize*2/3))
    ax11.grid(True, alpha=0.3, axis='y')
    
    # Jacobian vs features
    for i in range(min(5, jacobian_test.shape[1])):
        ax12.scatter(X_core_test[:, i], jacobian_test[:, i], 
                     alpha=0.3, s=10, label=f'Feature {i+1}')
    ax12.set_xlabel('Input Feature Value')
    ax12.set_ylabel('Jacobian (∂output/∂input)')
    ax12.set_title('Jacobian Values vs Input Features\n(Should be large but smooth)')
    ax12.legend(fontsize=int(fontsize*2/3))
    ax12.grid(True, alpha=0.3)

    all_axes = fig.axes
    for ax in all_axes:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize=fontsize)
    
    plt.tight_layout()
    plt.savefig('curriculum_dropout_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_predictions(ax, y_true, y_pred, mae, title, color):
    """Helper function to plot predictions vs true values"""
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color=color)
    ax.plot([y_true.min(), y_true.max()], 
            [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predictions')
    ax.set_title(f'{title} MAE: {mae:.4f}')
    ax.grid(True, alpha=0.3)


def plot_comparison_bars(ax, mae_without, mae_with, title):
    """Helper function to plot comparison bars"""
    ax.bar(['Without Aux', 'With Aux'], [mae_without, mae_with],
           color=['coral', 'skyblue'])
    ax.set_ylabel('MAE')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def generate_example_data(n_samples=1000, core_dim=5, aux_dim=3, seed=42):
    """Generate synthetic data for demonstration"""
    np.random.seed(seed)
    
    # Core features (always available)
    X_core = np.random.randn(n_samples, core_dim)
    
    # Auxiliary features (only available during training)
    X_aux = np.random.randn(n_samples, aux_dim)
    
    # Target: based primarily on core features with some influence from auxiliary
    y = (X_core[:, 0] * 2 + X_core[:, 1] - X_core[:, 2] * 0.5 + 
         X_aux[:, 0] * 0.3 + X_aux[:, 1] * 0.2 + 
         np.random.randn(n_samples) * 0.1)
    
    # Split data
    train_size = int(0.8 * n_samples)
    return {
        'X_core_train': X_core[:train_size],
        'X_core_test': X_core[train_size:],
        'X_aux_train': X_aux[:train_size],
        'X_aux_test': X_aux[train_size:],
        'y_train': y[:train_size],
        'y_test': y[train_size:]
    }


if __name__ == "__main__":
    # Example 1: Using the helper function with your data format
    print("="*60)
    print("EXAMPLE: Using prepare_train_test_split helper function")
    print("="*60)
    
    # Your data format
    data = {
        'X_core_train': np.random.randn(1000, 8),  # Replace with your core_data
        'X_aux_train': np.random.randn(1000, 3),   # Replace with your aux_data
        'y_train': np.random.randn(1000, 6)        # Replace with your output_data
    }
    
    # Split into train and test
    split_data = prepare_train_test_split(data, test_size=0.2, random_state=42)
    
    # Now you have all the data you need:
    # split_data['X_core_train'], split_data['X_core_test']
    # split_data['X_aux_train'], split_data['X_aux_test']
    # split_data['y_train'], split_data['y_test']
    
    # Define model architecture
    core_architecture = [
        {'units': 16, 'activation': 'relu'},
        {'units': 8, 'activation': 'relu'}
    ]
    
    aux_architecture = [
        {'units': 8, 'activation': 'relu'}
    ]
    
    combined_architecture = [
        {'units': 16, 'activation': 'relu'},
        {'units': 6}  # Output layer (6 outputs to match your data)
    ]
    
    # Build model
    print("\nBuilding model...")
    model, dropout_layer = build_auxiliary_dropout_model(
        core_input_dim=8,  # Match your data
        aux_input_dim=3,   # Match your data
        core_architecture=core_architecture,
        aux_architecture=aux_architecture,
        combined_architecture=combined_architecture,
        jacobian_weight=0.01,
        sv_weight=0.01
    )
    
    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
    
    # Train with curriculum learning
    print("\nTraining with curriculum learning...")
    history, curriculum_callback, X_core_val, X_aux_val, y_val = train_with_curriculum(
        model=model,
        dropout_layer=dropout_layer,
        X_core_train=split_data['X_core_train'],
        X_aux_train=split_data['X_aux_train'],
        y_train=split_data['y_train'],
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        initial_dropout=0.1,
        final_dropout=0.9,
        warmup_fraction=0.8,  # Reach final dropout at 80% of training
        verbose=0
    )
    
    # Evaluate on validation and test sets
    val_results = evaluate_model(model, X_core_val, X_aux_val, y_val, "Validation")
    test_results = evaluate_model(model, split_data['X_core_test'], 
                                   split_data['X_aux_test'], 
                                   split_data['y_test'], "Test")
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(
        history=history,
        curriculum_callback=curriculum_callback,
        val_results=val_results,
        test_results=test_results,
        y_val=y_val,
        y_test=split_data['y_test'],
        model=model,
        X_core_test=split_data['X_core_test'],
        X_aux_test=split_data['X_aux_test']
    )
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Final dropout rate: {curriculum_callback.dropout_history[-1]:.2f}")
    print("\nTraining complete! Check 'curriculum_dropout_results.png' for visualizations.")
