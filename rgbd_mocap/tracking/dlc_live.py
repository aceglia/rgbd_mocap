import numpy as np
try:
    from dlclive import DLCLive, Processor
except:
    raise ImportError("Please install the dlclive package to use the DeepLabCup model.")


class DlcLive:
    def __init__(self, dlc_model, dlc_processor=None, depth_scale=None, p_cutoff=0.6):
        self.depth_scale = 0.001 if depth_scale is None else depth_scale
        self.min_depth = None
        self.bg_remover_threshold = 8
        self.p_cutoff = p_cutoff
        self.value = None
        dlc_processor = Processor() if dlc_processor is None else dlc_processor
        self.dlc_live = DLCLive(dlc_model, processor=dlc_processor)
        self.inference_initialized = False
        self.last_value = None
        self.from_last_threshold = 50

    def get_pose(self, depth_frame, depth_scale=None, min_depth=None, bg_remover_threshold=None):
        self.depth_scale = depth_scale if depth_scale else self.depth_scale
        self.min_depth = min_depth if min_depth else self.min_depth
        self.bg_remover_threshold = bg_remover_threshold if bg_remover_threshold else self.bg_remover_threshold
        depth_frame = self._process_depth(depth_frame, self.depth_scale, self.min_depth, self.bg_remover_threshold)
        if not self.inference_initialized:
            self. inference_initialized = True
            return self.dlc_live.init_inference(depth_frame)
        self.last_value = self.value.copy()
        self.value = self.dlc_live.get_pose(depth_frame)
        return self.value

    def __getitem__(self, item):
        if self.value is not None:
            return self.value[item, :2], self.value[item, 2]
        else:
            raise ValueError("Please run the get_pose method before accessing the values.")

    @staticmethod
    def _process_depth(depth_frame, depth_scale, min_depth, bg_remover_threshold):
        depth = np.where((depth_frame > bg_remover_threshold / depth_scale) | (depth_frame <= 0), 0, depth_frame)
        min_depth = min_depth if min_depth else np.min(depth[depth > 0])
        max_depth = np.max(depth)
        normalize_depth = (depth - min_depth) / (max_depth - min_depth)
        normalize_depth[depth == 0] = 0
        normalize_depth = normalize_depth * 255
        depth = normalize_depth.astype(np.uint8)
        return np.dstack((depth, depth, depth))

    def check_from_last(self):
        for i in range(len(self.last_value)):
            dist = np.linalg.norm(self.last_value[i, :2] - self.value[i, :2])
            if dist > self.from_last_threshold:
                self.value[i] = self.last_value[i]
