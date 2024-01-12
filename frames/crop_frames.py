from frames.frames import Frames


class CropFrames:
    def __init__(self, area, frame: Frames):
        self.area = area

        self.frame = frame
        self.color, self.depth = frame.get_crop(area)

        self.width = self.color.shape[0]
        self.height = self.depth.shape[1]

    def _update_image(self):
        self.color, self.depth = self.frame.get_crop(self.area)

    def get_image(self):
        self._update_image()
        return self.color, self.depth
