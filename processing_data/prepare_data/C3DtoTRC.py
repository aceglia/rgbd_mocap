import numpy as np

try:
    from pyomeca import Markers
except:
    pass
import csv
import glob


class WriteTrc:
    def __init__(self):
        self.output_file_path = None
        self.input_file_path = None
        self.markers = None
        self.marker_names = None
        self.data_rate = None
        self.cam_rate = None
        self.n_frames = None
        self.start_frame = None
        self.units = None
        self.channels = None
        self.time = None

    def _prepare_trc(self):
        headers = [
            ["PathFileType", 4, "(X/Y/Z)", self.output_file_path],
            [
                "DataRate",
                "CameraRate",
                "NumFrames",
                "NumMarkers",
                "Units",
                "OrigDataRate",
                "OrigDataStartFrame",
                "OrigNumFrames",
            ],
            [
                self.data_rate,
                self.cam_rate,
                self.n_frames,
                len(self.marker_names),
                self.units,
                self.data_rate,
                self.start_frame,
                self.n_frames,
            ],
        ]
        markers_row = [
            "Frame#",
            "Time",
        ]
        coord_row = ["", ""]
        empty_row = []
        idx = 0
        for i in range(len(self.marker_names) * 3):
            if i % 3 == 0:
                markers_row.append(self.marker_names[idx])
                idx += 1
            else:
                markers_row.append(None)
        headers.append(markers_row)

        for i in range(len(self.marker_names)):
            name_coord = 0
            while name_coord < 3:
                if name_coord == 0:
                    coord_row.append(f"X{i+1}")
                elif name_coord == 1:
                    coord_row.append(f"Y{i+1}")
                elif name_coord == 2:
                    coord_row.append(f"Z{i+1}")
                name_coord += 1

        headers.append(coord_row)
        headers.append(empty_row)
        return headers

    def write_trc(self):
        if self.input_file_path:
            self._read_c3d()
        headers = self._prepare_trc()
        duration = self.n_frames / self.data_rate
        time = np.around(np.linspace(0, duration, self.n_frames), decimals=3) if self.time is None else self.time
        for frame in range(self.markers.shape[2]):
            row = [frame + 1, time[frame]]
            for i in range(self.markers.shape[1]):
                for j in range(3):
                    row.append(self.markers[j, i, frame])
            headers.append(row)
        with open(self.output_file_path, "w", newline="") as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerows(headers)

    def _read_c3d(self):
        data = Markers.from_c3d(self.input_file_path, usecols=self.channels)
        self.markers = data.values[:3, :, :]
        self.n_frames = len(data.time.values)
        self.marker_names = data.channel.values.tolist()
        self.data_rate = data.attrs["rate"]
        self.units = data.attrs["units"]
        self.start_frame = data.attrs["first_frame"] + 1
        self.cam_rate = self.data_rate if self.cam_rate is None else self.cam_rate
        self.time = data.time.values


class WriteTrcFromC3d(WriteTrc):
    def __init__(
        self,
        output_file_path,
        c3d_file_path,
        data_rate=None,
        cam_rate=None,
        n_frames=None,
        start_frame=1,
        c3d_channels=None,
    ):
        super(WriteTrcFromC3d, self).__init__()
        self.input_file_path = c3d_file_path
        self.output_file_path = output_file_path
        self.data_rate = data_rate
        self.cam_rate = self.data_rate if cam_rate is None else cam_rate
        self.n_frames = n_frames
        self.start_frame = start_frame
        self.channels = c3d_channels

    def write(self):
        self.write_trc()


class WriteTrcFromMarkersData(WriteTrc):
    def __init__(
        self,
        output_file_path,
        markers=None,
        marker_names=None,
        data_rate=None,
        cam_rate=None,
        n_frames=None,
        start_frame=1,
        units="m",
    ):
        super(WriteTrcFromMarkersData, self).__init__()
        self.output_file_path = output_file_path
        self.markers = markers
        self.marker_names = marker_names
        self.data_rate = data_rate
        self.cam_rate = self.data_rate if cam_rate is None else cam_rate
        self.n_frames = n_frames
        self.start_frame = start_frame
        self.units = units
        self.time = None

    def write(self):
        self.write_trc()


if __name__ == "__main__":
    import pathlib

    # outfile_path = "data.trc"
    # infile_path = "data.c3d"
    from biosiglive import load

    # markers_file = "data_files/P3_session2/gear_20_15-08-2023_09_35_38/P3_gear_20_c3d.bio"
    # data = load(markers_file)
    # markers_name = ["STER",
    #                  "XIPH",
    #                  "C7",
    #                  "T5",
    #                  'RIBS_r',
    #                  "CLAV_SC",
    #                  "CLAV_AC",
    #                  "SCAP_TS",
    #                  "SCAP_IA",
    #                  "SCAP_AA",
    #                  "DELT",
    #                  "ARMl",
    #                  "EPICM",
    #                  "EPICl",
    #                  "ELB",
    #                  "larm_l",
    #                  "STYLr",
    #                  "STYLu"
    #                 ]
    #
    # # markers_name = markers_data["markers"].channel.values.tolist()[:-5]
    # markers_data = data["markers"].values[:3, :-5, :200].round(5) * 0.001
    # outfile_path = "data_files/P4_session2/gear_20_15-08-2023_09_35_38/P3_gear_20_c3d.trc"
    # write = WriteTrcFromMarkersData(outfile_path,
    #                         markers_data,
    #                         markers_name,
    #                         120,
    #                         120,
    #                         markers_data.shape[2],
    #                         units="m"
    #                         )
    # write.time = data["markers"].time.values.round(4)
    # write.write()
    participant = "P4_session2"
    trial = "standing_anato"
    file_dir = rf"data_files\{participant}\standing_anato_15-08-2023_10_28_27"
    vicon_data = load(rf"{file_dir}\P4_{trial}_c3d.bio")
    import glob

    # markers_depth = load(f"{file_dir}\markers_kalman.bio", number_of_line=len(glob.glob(file_dir + "\color*")))
    # # end = markers_depth.shape[2]
    # markers_depth = markers_depth["markers_in_meters"]
    # markers_depth_names = [
    #                  "T5",
    #                  "C7",
    #                  'RIBS_r',
    #                  "CLAV_AC",
    #                  "SCAP_TS",
    #                  "SCAP_IA",
    #                  "SCAP_AA",
    #                  "DELT",
    #                  "ARMl",
    #                  "EPICl",
    #                  "larm_l",
    #                  "STYLr",
    #                  "STYLu"
    #                 ]
    markers_vicon_names = [
        "STER",
        "XIPH",
        "C7",
        "T5",
        "RIBS_r",
        "CLAV_SC",
        "CLAV_AC",
        "SCAP_TS",
        "SCAP_IA",
        "SCAP_AA",
        "DELT",
        "ARMl",
        "EPICM",
        "EPICl",
        "ELB",
        "larm_l",
        "STYLr",
        "STYLu",
    ]
    # # new_markers_depth = np.zeros((3, markers_depth.shape[1], idx[-1]-idx[0]))
    # count = 0
    # # for i in range(idx[-1]-idx[0]):
    # #     if i + start_color_idx in idx:
    # #         new_markers_depth[:, :, i] = np.dot(np.array(optimal_rotation),
    # #                                             np.array(markers_depth[:, :, count])
    # #                                             ) + np.array(optimal_translation)
    # #         count += 1
    # #     else:
    # #         new_markers_depth[:, :, i] = np.nan
    markers_vicon = vicon_data["markers"].values[:3, : len(markers_vicon_names), :] * 0.001
    # WriteTrcFromMarkersData(output_file_path =f"{file_dir}\{participant}_{trial}_from_depth.trc",
    #                         markers=np.round(markers_depth, 5),
    #                         marker_names=markers_depth_names,
    #                         data_rate=60,
    #                         cam_rate=60,
    #                         n_frames=markers_depth.shape[2],
    #                         start_frame=1,
    #                         units="m").write()

    WriteTrcFromMarkersData(
        output_file_path=f"{file_dir}\{participant}_{trial}_from_vicon.trc",
        markers=np.round(markers_vicon, 5),
        marker_names=markers_vicon_names,
        data_rate=120,
        cam_rate=120,
        n_frames=markers_vicon.shape[2],
        start_frame=1,
        units="m",
    ).write()
