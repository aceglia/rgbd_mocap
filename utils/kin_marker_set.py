from rgbd_mocap.marker_class import MarkerSet


class KinMarkerSet:
    def __new__(cls, camera, option_from):
        ### Init the marker_sets and kin_marker_set
        marker_sets, kin_marker_set = option_from()

        ### Add the marker_sets to the given camera
        camera.add_marker_set(marker_sets)

        ### Return the kin_marker_set
        return kin_marker_set

    @staticmethod
    def from_back_3():
        # ----------- from back ---------------- #
        shoulder_list = ["T5", "C7", "RIBS_r", "Clavsc", "Scap_AA", "Scap_IA", "Acrom"]
        arm_list = ["delt", "arm_l", "epic_l"]
        hand_list = ['larm_l', "styl_r", "styl_u"]

        ### MarkerSets
        marker_sets = KinMarkerSet.create_set(['shoulder', 'arm', 'hand'],
                                              [shoulder_list, arm_list, hand_list],
                                              image_index=True)

        ### KinematicsMarkerSets
        kin_marker_sets = KinMarkerSet.create_set(['shoulder', 'scapula', 'arm', 'hand'],
                                                  [shoulder_list[:4], shoulder_list[4:], arm_list, hand_list])

        return marker_sets, kin_marker_sets

    @staticmethod
    def from_front_3():
        # ------------------ from front 3 crops -----------------#
        shoulder_list = ["xiph", "ster", "clavsc", "M1", "M2", "M3", "Clavac", ]
        arm_list = ["delt", "arm_l", "epic_l"]
        hand_list = ["larm_l", "styl_r", "styl_u"]

        ### MarkerSets
        ### index will be 0, 1, 2 and not 0, 2, 3 (does it cause a problem ?)
        marker_sets = KinMarkerSet.create_set(['shoulder', 'arm', 'hand'],
                                              [shoulder_list, arm_list, hand_list],
                                              image_index=True)

        ### KinematicsMarkerSets
        kin_marker_sets = KinMarkerSet.create_set(['shoulder', 'arm', 'hand'],
                                                  [shoulder_list, arm_list, hand_list])

        return marker_sets, kin_marker_sets

    @staticmethod
    def from_front_4():
        # ------------------ from front 4 crops -----------------#
        thorax_list = ["xiph", "ster", "clavsc"]
        cluster_list = ["M1", "M2", "M3", "Clavac"]
        arm_list = ["delt", "arm_l", "epic_l"]
        hand_list = ["larm_l", "styl_r", "styl_u"]

        ### MarkerSets
        marker_sets = KinMarkerSet.create_set(['thorax', 'cluster', 'arm', 'hand'],
                                              [thorax_list, cluster_list, arm_list, hand_list],
                                              image_index=True)

        ### KinematicsMarkerSets
        kin_marker_sets = KinMarkerSet.create_set(['shoulder', 'arm', 'hand'],
                                                  [thorax_list + cluster_list, arm_list, hand_list])

        return marker_sets, kin_marker_sets

    @staticmethod
    def create_set(set_names, set_list, image_index=False):
        marker_set = []

        for i in range(len(set_names)):
            marker_set.append(MarkerSet(marker_set_name=set_names[i],
                                        marker_names=set_list[i],
                                        image_idx=i * image_index))  # will always be 0 if image_index is False

        return marker_set

    ### Link Class enum to the corresponding methods
    BACK_3 = from_back_3
    FRONT_3 = from_front_3
    FRONT_4 = from_front_4
