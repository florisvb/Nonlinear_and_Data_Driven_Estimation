import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# This function was written by Ben Cellini, and subsequently expanded by Claude to provide further functionality
def collect_offset_rows(df, states=None, controls=None, outputs=None,
                        state_offsets=None, control_offsets=None, output_offsets=None):
    """ Takes a pandas data frame with n rows and creates an augmented data frame that collects 
        rows at specified offsets for different column types (states, controls, outputs).
        
        Inputs
            df: pandas data frame
            states: list of column names to treat as states, or primary inputs
            controls: list of column names to treat as controls, or auxiliary inputs
            outputs: list of column names to treat as outputs
            state_offsets: list of integer offsets for state columns (default: [0])
            control_offsets: list of integer offsets for control columns (default: [0])
            output_offsets: list of integer offsets for output columns (default: [0])
            
            Offset interpretation:
                - Negative values look backward (e.g., -1 is previous row, -2 is two rows back)
                - Zero includes current row
                - Positive values look forward (e.g., 1 is next row, 2 is two rows ahead)
                
        Outputs
            df_aug: augmented pandas data frame.
                    Columns are organized by type and offset:
                    - First all state columns (grouped by offset)
                    - Then all control columns (grouped by offset)
                    - Finally all output columns (grouped by offset)
                    New columns are named: old_name_offset_N (e.g., 'x_offset_-1', 'u_offset_0')
                    Only rows where all offsets are valid are included.
    """
    import pandas as pd
    import numpy as np
    df = df.reset_index(drop=True)
    
    # Set defaults
    if state_offsets is None:
        state_offsets = [-1]
    if control_offsets is None:
        control_offsets = [-1]
    if output_offsets is None:
        output_offsets = [0]
    
    # Collect all columns and offsets to determine valid row range
    all_columns = []
    all_offsets = []
    
    column_groups = []
    if states is not None and len(states) > 0:
        all_columns.extend(states)
        all_offsets.extend(state_offsets)
        column_groups.append(('states', states, state_offsets))
    
    if controls is not None and len(controls) > 0:
        all_columns.extend(controls)
        all_offsets.extend(control_offsets)
        column_groups.append(('controls', controls, control_offsets))
    
    if outputs is not None and len(outputs) > 0:
        all_columns.extend(outputs)
        all_offsets.extend(output_offsets)
        column_groups.append(('outputs', outputs, output_offsets))
    
    if len(all_columns) == 0:
        raise ValueError("At least one of states, controls, or outputs must be specified")
    
    # Determine valid row range
    min_offset = min(all_offsets)
    max_offset = max(all_offsets)
    
    n_row = df.shape[0]
    
    # Valid rows are those where all offsets point to existing rows
    start_row = max(0, -min_offset)
    end_row = min(n_row, n_row - max_offset)
    n_row_train = end_row - start_row
    
    if n_row_train <= 0:
        raise ValueError(f"No valid rows with given offsets. DataFrame has {n_row} rows, "
                        f"but offsets range from {min_offset} to {max_offset}")
    
    # Build augmented dataframe by processing each column group
    df_aug_parts = []
    
    for group_name, columns, offsets in column_groups:
        # For each offset in this group
        for offset in offsets:
            source_indices = np.arange(start_row, end_row) + offset
            
            # For each column with this offset
            for col in columns:
                data = df.loc[source_indices, col].reset_index(drop=True)
                col_name = f'{col}_offset_{offset}'
                df_aug_parts.append(pd.DataFrame({col_name: data}))
    
    # Combine all parts
    df_aug = pd.concat(df_aug_parts, axis=1)
    
    return df_aug

# almost direct from Claude
def create_fast_inference_model(trained_model):
    """
    Create an optimized version of the model for fast inference using XLA compilation.
    Removes training-specific components (dropout, regularization).
    
    Note: The first prediction will be slower as it includes compilation time.
    Subsequent predictions will be fast.
    
    Parameters:
    -----------
    trained_model : keras model
        The trained model
    
    Returns:
    --------
    inference_func : tf.function
        Optimized inference function for predictions
    
    Example:
    --------
    >>> fast_predict = create_fast_inference_model(model)
    >>> # First call is slow (compilation)
    >>> predictions = fast_predict(X_core_batch, X_aux_batch) # X_aux_batch is optional, depending on model
    >>> # Subsequent calls are fast
    >>> predictions = fast_predict(X_core_batch2, X_aux_batch2) # X_aux_batch is optional, depending on model
    """
    # Get the base functional model (without custom training logic)
    inputs = trained_model.inputs
    outputs = trained_model.outputs
    
    # Create inference-only model
    inference_model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Get only trainable weights (excludes metric trackers)
    trainable_weights = trained_model.trainable_weights
    inference_trainable_weights = inference_model.trainable_weights
    
    # Copy weights
    for target_weight, source_weight in zip(inference_trainable_weights, trainable_weights):
        target_weight.assign(source_weight)
    
    # Create optimized prediction function
    @tf.function(jit_compile=True)  # XLA compilation for speed
    def fast_predict(x_core, x_aux=None):
        if x_aux is None:
            return inference_model([x_core], training=False)
        else:
            return inference_model([x_core, x_aux], training=False)
    
    print("✓ Fast inference model created with XLA compilation")
    print("  First prediction will be slow (includes compilation)")
    print("  Subsequent predictions will be fast")
    print("  Use: predictions = fast_predict(X_core_tensor, X_aux_tensor)")
    print("  Note: Inputs must be TensorFlow tensors (use tf.constant())")
    
    return fast_predict