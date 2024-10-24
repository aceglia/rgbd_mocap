import numpy as np
import cv2

try:
    from dlclive import DLCLive, Processor
except:
    raise ImportError("Please install the dlclive package to use the DeepLabCup model.")


class DlcLive:
    def __init__(self, dlc_model, dlc_processor=None, depth_scale=None, p_cutoff=0, downsample_ratio=None):
        self.depth_scale = 0.001 if depth_scale is None else depth_scale
        self.min_depth = None
        self.bg_remover_threshold = 8
        self.p_cutoff = p_cutoff
        self.value = None
        self.model = dlc_model
        dlc_processor = Processor() if dlc_processor is None else dlc_processor
        self.dlc_live = DLCLive(dlc_model, processor=dlc_processor)
        self.inference_initialized = False
        self.last_value = None
        self.from_last_threshold = 80
        self.max_depth = -1
        self.min_depth = -1
        self.depth_image = None
        self.normal = None
        self.downsample_ratio = downsample_ratio

    def get_pose(self, depth_frame=None, depth_scale=None, min_depth=None, bg_remover_threshold=None):
        import time

        self.depth_scale = depth_scale if depth_scale else self.depth_scale
        self.min_depth = min_depth if min_depth else self.min_depth
        self.bg_remover_threshold = bg_remover_threshold if bg_remover_threshold else self.bg_remover_threshold
        depth_frame = self.depth_image if depth_frame is None else self._process_depth(depth_frame, self.depth_scale)
        if not self.inference_initialized:
            self.inference_initialized = True
            pos = self.dlc_live.init_inference(depth_frame)
            return pos

        self.last_value = None if self.value is None else self.value.copy()

        tic = time.time()
        self.value = self.dlc_live.get_pose(depth_frame)
        print("time to get dlc =", time.time() - tic)
        self.last_value = self.value.copy() if self.last_value is None else self.last_value
        return self.value

    def __getitem__(self, item):
        if self.value is not None:
            return self.value[item, :2], self.value[item, 2]
        else:
            raise ValueError("Please run the get_pose method before accessing the values.")

    def update_depth_frame(self, depth_frame):
        import time

        tic = time.time()
        self.depth_image = self._process_depth(depth_frame, self.depth_scale)
        cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
        cv2.imshow("depth", self.depth_image)
        cv2.waitKey(1)
        print("time to process =", time.time() - tic)

    def _process_depth(self, depth_frame, depth_scale):
        self.bg_remover_threshold = 1.2
        depth = np.where(
            (depth_frame > self.bg_remover_threshold / depth_scale) | (depth_frame <= 0.2 / (0.0010000000474974513)),
            0,
            depth_frame,
        )
        return self.compute_surface_normals(depth)

    #
    def compute_surface_normals(self, depth_map):
        dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1)

        if self.normal is None or self.normal.shape != depth_map.shape:
            self.normal = np.empty((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.float32)

        self.normal[..., 0] = -dx
        self.normal[..., 1] = -dy
        self.normal[..., 2] = 1.0

        self.normal /= np.linalg.norm(self.normal, axis=2, keepdims=True)
        # Map the normal vectors to the [0, 255] range and convert to uint8
        self.normal = (self.normal + 1.0) * 127.5
        self.normal = np.clip(self.normal, 0, 255).astype(np.uint8)
        # Convert normal to BGR format for visualization (assuming RGB input)
        self.normal = cv2.cvtColor(self.normal, cv2.COLOR_RGB2BGR)
        # normal_bgr = normal
        return self.normal

    def check_from_last(self):
        for i in range(len(self.last_value)):
            dist = np.linalg.norm(self.last_value[i, :2] - self.value[i, :2])
            if dist > self.from_last_threshold:
                self.value[i] = self.last_value[i]
