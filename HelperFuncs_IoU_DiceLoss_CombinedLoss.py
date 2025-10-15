import tensorflow as tf

def mean_iou(y_true, y_pred):
    """Calculates the Mean Intersection over Union metric."""
    y_pred = tf.argmax(y_pred, axis=-1)
    
    # Calculate the confusion matrix
    cm = tf.math.confusion_matrix(tf.cast(tf.reshape(y_true, [-1]), tf.int32), 
                                  tf.cast(tf.reshape(y_pred, [-1]), tf.int32), 
                                  num_classes=tf.shape(y_pred)[-1] + 1) # Assuming num_classes can be inferred
    
    # Calculate IoU for each class
    intersection = tf.linalg.tensor_diag_part(cm)
    union = tf.reduce_sum(cm, axis=1) + tf.reduce_sum(cm, axis=0) - intersection
    
    # Avoid division by zero
    iou = tf.math.divide_no_nan(intersection, union)
    
    # Return mean IoU
    return tf.reduce_mean(iou)

def dice_loss(y_true, y_pred, smooth=1e-6):
    """Computes Dice loss for sparse labels."""
    num_classes = tf.shape(y_pred)[-1]
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
    y_true_squeezed = tf.squeeze(y_true_one_hot, axis=-2)
    
    y_true_f = tf.reshape(y_true_squeezed, [-1, num_classes])
    y_pred_f = tf.reshape(y_pred, [-1, num_classes])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    denominator = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)
    
    dice = (2. * intersection + smooth) / (denominator + smooth)
    return 1 - tf.reduce_mean(dice)

def combined_loss(y_true, y_pred):
    """Combines Dice loss and Sparse Categorical Crossentropy."""
    sce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    return 0.5 * dice_loss(y_true, y_pred) + 0.5 * sce_loss(y_true, y_pred)