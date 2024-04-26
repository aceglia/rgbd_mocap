import cv2

from ..tracking.utils import set_marker_pos
from ..processing.handler import Handler


class ProcessHandler(Handler):
    def __init__(self, crops):
        super().__init__()
        self.crops = crops
        self.crops_name = [crop.marker_set.name for crop in self.crops]
        self.tracking_option = self.crops[0].tracking_option
        self.blobs = []
        print(self.crops_name)

    def _process_function(self, order):
        self.blobs = []
        if order == Handler.CONTINUE:
            for i, crop in enumerate(self.crops):
                blobs, positions, estimate_positions = crop.track_markers()
                set_marker_pos(crop.marker_set, positions)
                Handler.show_image(f"{crop.marker_set.name}",
                                   crop.filter.filtered_frame,
                                   blobs=blobs,
                                   markers=crop.marker_set,
                                   estimated_positions=estimate_positions)
                self.blobs += blobs

        elif order == Handler.RESET:
            for crop in self.crops:
                crop.re_init(crop.marker_set, self.tracking_option)

    def send_process(self, order=1):
        self._process_function(order)

    def send_and_receive_process(self, order=1):
        self.send_process(order)
