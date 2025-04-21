import numpy as np
import ctypes
import time

# 加载 CUDA 共享库
lib = ctypes.CDLL('./cuda_op/vector_add.dll')

# 设置函数参数类型
lib.vector_add_cuda.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # float *a
    ctypes.POINTER(ctypes.c_float),  # float *b
    ctypes.POINTER(ctypes.c_float),  # float *c
    ctypes.c_int                     # int n
]

# 测试代码
def test_vector_add():
    import time
    # 创建测试数据
    n = 1024 * 1024 * 50  # 向量长度
    a = np.random.random(n).astype(np.float32)
    b = np.random.random(n).astype(np.float32)
    c = np.zeros(n, dtype=np.float32)

    start_time = time.time()
    # 获取指向数组的指针
    a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_ptr = c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # 调用 CUDA 函数
    lib.vector_add_cuda(a_ptr, b_ptr, c_ptr, n)
    
    time = time.time()-start_time

    # 验证结果
    np.testing.assert_almost_equal(c, a + b, decimal=5)
    print("Vector addition completed successfully!")

    return time

if __name__ == "__main__":
    n = 1024*1024*50
    time1 = test_vector_add()
    a = np.random.random(n).astype(np.float32)
    b = np.random.random(n).astype(np.float32)
    start_time = time.time()
    c = a+b
    time2 = time.time()-start_time

    a = np.random.random(n).astype(np.float32).tolist()
    b = np.random.random(n).astype(np.float32).tolist()
    start_time = time.time()
    c = a+b
    time3 = time.time()-start_time

    print(time1, time2, time3)
