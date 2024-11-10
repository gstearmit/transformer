import tensorflow as tf

# Định nghĩa hàm mất mát sử dụng SparseCategoricalCrossentropy
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    # Hàm tính mất mát cho mô hình dịch máy
    
    # Tạo mask để loại bỏ padding (giá trị 0)
    mask = tf.math.logical_not(real == 0)
    
    # Tính mất mát sử dụng hàm loss_object
    loss = loss_object(real, pred)

    # Chuyển đổi mask sang cùng kiểu dữ liệu với loss
    mask = tf.cast(mask, dtype=loss.dtype)
    
    # Áp dụng mask vào loss để bỏ qua các giá trị padding
    loss = loss * mask

    # Tính trung bình loss, chỉ tính trên các phần tử không phải padding
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)