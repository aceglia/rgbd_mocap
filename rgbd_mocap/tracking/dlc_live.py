import numpy as np
import cv2
try:
    from dlclive import DLCLive, Processor
except:
    raise ImportError("Please install the dlclive package to use the DeepLabCup model.")


class DlcLive:
    def __init__(self, dlc_model, dlc_processor=None, depth_scale=None, p_cutoff=0):
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

    def get_pose(self, depth_frame=None, depth_scale=None, min_depth=None, bg_remover_threshold=None):
        self.depth_scale = depth_scale if depth_scale else self.depth_scale
        self.min_depth = min_depth if min_depth else self.min_depth
        self.bg_remover_threshold = bg_remover_threshold if bg_remover_threshold else self.bg_remover_threshold
        depth_frame = self.depth_image if depth_frame is None else self._process_depth(depth_frame, self.depth_scale)

        if not self.inference_initialized:
            self. inference_initialized = True
            pos = self.dlc_live.init_inference(depth_frame)
            return pos
        self.last_value = None if self.value is None else self.value.copy()
        self.value = self.dlc_live.get_pose(depth_frame)
        self.last_value = self.value.copy() if self.last_value is None else self.last_value
        return self.value

    def __getitem__(self, item):
        if self.value is not None:
            return self.value[item, :2], self.value[item, 2]
        else:
            raise ValueError("Please run the get_pose method before accessing the values.")

    def update_depth_frame(self, depth_frame):
        self.depth_image = self._process_depth(depth_frame, self.depth_scale)

    def _process_depth(self, depth_frame, depth_scale):
        self.bg_remover_threshold = 1.2
        depth = np.where((depth_frame > self.bg_remover_threshold / depth_scale) | (depth_frame <=  0.2 / (0.0010000000474974513)), 0, depth_frame)

        return self.compute_surface_normals(depth)
        # self.max_depth = self.max_depth if self.max_depth != -1 else np.median(np.sort(depth.flatten())[-30:])
        # self.min_depth = 0.4 / (0.0010000000474974513)
        # # min_depth = min_depth if min_depth else np.min(depth[depth > 0])
        # # max_depth = np.max(depth)
        # normalize_depth = (depth - self.min_depth) / (self.max_depth - self.min_depth)
        # normalize_depth[depth == 0] = 0
        # normalize_depth = normalize_depth * 255
        # depth = normalize_depth.astype(np.uint8)
        # depth_3d = np.dstack((depth, depth, depth))
        # kernel = np.array([[-1, 0, -1],
        #                    [-1, 7, -1],
        #                    [-1, 0, -1]])
        # if "hist" in self.model or "colormap" in self.model:
        #     hist_eq = cv2.equalizeHist(depth)
        #     hist_3d = np.dstack((hist_eq, hist_eq, hist_eq))
        #     depth_colormap = cv2.applyColorMap(hist_3d, cv2.COLORMAP_JET)
        #     depth_colormap[depth_3d == 0] = 0
        #     depth_colormap =  cv2.filter2D(depth_colormap, -1,
        #                         kernel)
        #
        #     return depth_colormap
        #     if "sharp" not in self.model:
        #         return hist_3d
        #     elif "colormap" not in self.model:
        #         return cv2.filter2D(hist_3d, -1,
        #                                        kernel)
        #     else:
        #         depth_colormap = cv2.applyColorMap(hist_3d, cv2.COLORMAP_JET)
        #         depth_colormap[depth_3d == 0] = 0
        #         return depth_colormap
        # else:
        #     raise RuntimeError("model not recognize")
        # # return depth_3d
    #
    @staticmethod
    def compute_surface_normals(depth_map):
        rows, cols = depth_map.shape

        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        # Calculate the partial derivatives of depth with respect to x and y
        dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1)

        # Compute the normal vector for each pixel
        normal = np.dstack((-dx, -dy, np.ones((rows, cols))))
        norm = np.sqrt(np.sum(normal ** 2, axis=2, keepdims=True))
        normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm != 0)

        # Map the normal vectors to the [0, 255] range and convert to uint8
        normal = (normal + 1) * 127.5
        normal = normal.clip(0, 255).astype(np.uint8)

        # Save the normal map to a file
        normal_bgr = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
        return normal_bgr

    def check_from_last(self):
        for i in range(len(self.last_value)):
            dist = np.linalg.norm(self.last_value[i, :2] - self.value[i, :2])
            if dist > self.from_last_threshold:
                self.value[i] = self.last_value[i]
