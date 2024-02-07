from ..markers.marker_set import MarkerSet
from ..frames.frames import Frames
from ..crop.crop import Crop
from ..tracking.test_tracking import set_marker_pos
from ..processing.handler import Handler


class ProcessHandler(Handler):
    def __init__(self, markers_sets: list[MarkerSet], frames: Frames, options, tracking_option):
        super().__init__()
        self.crops = []
        self.crops_name = [crop['name'] for crop in options['crops']]
        self.tracking_option = tracking_option

        print(self.crops_name)

        for i in range(len(markers_sets)):
            marker_set = markers_sets[i]
            option = options['crops'][i]

            # Init Crop
            crop = Crop(option['area'], frames, marker_set, option['filters'], tracking_option)

            self.crops.append(crop)

    def _process_function(self, order):
        if order == Handler.CONTINUE:
            for i, crop in enumerate(self.crops):
                blobs, positions, estimate_positions = crop.track_markers()
                set_marker_pos(crop.marker_set, positions)

                self.show_image(f"{self.crops_name[i]}",
                                crop.filter.filtered_frame,
                                blobs=blobs,
                                markers=crop.marker_set,
                                estimated_positions=estimate_positions)

        elif order == Handler.RESET:
            for crop in self.crops:
                crop.re_init(crop.marker_set, self.tracking_option)

    def send_process(self, order=1):
        self._process_function(order)

    def send_and_receive_process(self, order=1):
        self.send_process(order)
