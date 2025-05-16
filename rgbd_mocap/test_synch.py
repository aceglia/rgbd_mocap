import numpy as np
import time
from rgbd_mocap.frames.shared_frames import SharedFrames
from multiprocessing import RawArray, RawValue
import multiprocessing as mp


color_array = RawArray("c", (3*848*480*10))  # 'c' -> value between 0-255
color = np.frombuffer(color_array, dtype=np.uint8).reshape((3*848*480*10))


# depth_array = RawArray("i", self.width * self.height)  # 'i' -> int32
# index = RawValue("i", 0)
mat = np.ndarray((3, 848, 480, 10))
to_add = np.ndarray((3, 848, 480)).astype(np.uint8).flatten()

tic = time.time()
for i in range(1):
    # mat[..., 0] = to_add
    # np.roll(mat, axis=-1, shift=1)
    np.copyto(color[to_add.shape[0] * 0:to_add.shape[0]*1], to_add)

    # np.concatenate((mat[..., 1:], to_add[..., None]), axis=-1)
    # mat[..., :-1] = mat[..., 1:]
    # mat[..., -1] = to_add
np.allclose(color[to_add.shape[0] * 0:to_add.shape[0]*1].reshape((3, 848, 480)), to_add)
print("time for roll", time.time() - tic)