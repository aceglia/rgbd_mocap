from ..frames.frames import Frames


class CropFrames(Frames):
    def __init__(self, area, frame: Frames,):
        super().__init__(frame.color, frame.depth, frame.index)
        self.area = area

        self.frame = frame
        self.color, self.depth = frame.get_crop(area)

        self.width = self.color.shape[1]
        self.height = self.depth.shape[0]

    def update_image(self):
        self.color, self.depth = self.frame.get_crop(self.area)

    def get_index(self):
        return self.frame.get_index()

    def get_images(self):
        self.update_image()
        return self.color, self.depth
