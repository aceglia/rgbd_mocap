class MarkerSet:
    """
    This class is used to store the marker information
    """
    def __init__(self, marker_names: list, image_idx: int = 0):
        """
        init markers class with number of markers, names and image index

        Parameters
        ----------
        marker_names : list
            list of names for the markers
        image_idx : list
            index of the image where the marker set is located
        """
        self.nb_markers = len(marker_names)
        self.image_idx = image_idx
        self.marker_names = marker_names
        self.pos = np.zeros((2, self.nb_markers, 1))
        self.speed = np.zeros((2, self.nb_markers, 1))
        self.marker_set_model = None
        self.markers_idx_in_image = []
        self.estimated_area = []
        self.next_pos = np.zeros((2, self.nb_markers, 1))
        self.model = None

    @staticmethod
    def compute_speed(pos, pos_old, dt=1):
        """
        Compute the speed of the markers
        """
        return (pos - pos_old) / dt

    @staticmethod
    def compute_next_position(speed, pos, dt=1):
        """
        Compute the next position of the markers
        """
        return pos + speed * dt

    def update_speed(self):
        for i in range(2):
            self.speed[i, :] = self.compute_speed(self.pos[i, :, -1], self.pos[i, :, -2])

    def update_next_position(self):
        """
        Update the next position of the markers
        """
        next_pos=[]
        for i in range(2):
            # next_pos.append(np.concatenate((self.next_pos[i, :, :], self.compute_next_position(self.speed[i, :], self.pos[i, :, -1])[:, np.newaxis]), axis=1))
            self.next_pos[i, :] = self.compute_next_position(self.speed[i, :], self.pos[i, :, -1])[:, np.newaxis]
