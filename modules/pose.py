import numpy as np


_Nose = 0
_Neck = 1
_RShoulder = 2
_RElbow = 3
_RWrist = 4
_LShoulder = 5
_LElbow = 6
_LWrist = 7
_RHip = 8
_RKnee = 9
_RAnkle = 10
_LHip = 11
_LKnee = 12
_LAnkle = 13
_REye = 14
_LEye = 15
_REar = 16
_LEar = 17

body_kp_id_to_name = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar"
]

body_kp_name_to_id = {body_kp_id_to_name[i]: i for i in range(len(body_kp_id_to_name))}

body_edges = np.array(
    [[_Neck, _Nose], 
     [_Nose, _LEye], [_LEye, _LEar], 
     [_Nose, _REye], [_REye, _REar], 
     [_Neck, _LShoulder], [_LShoulder, _LElbow], [_LElbow, _LWrist],   
     [_Neck, _RShoulder], [_RShoulder, _RElbow], [_RElbow, _RWrist], 
     [_Neck, _LHip], [_LHip, _LKnee], [_LKnee, _LAnkle],  
     [_Neck, _RHip], [_RHip, _RKnee], [_RKnee, _RAnkle]])

class Pose:
    def __init__(self, array_55):
        """
        array_55 is a flat array with 54 elements represents 18 keypoint information (x, y, conf)
        and 55th element is global conf
        """
        self.array_55 = array_55

    def get_body_kp(self, kp_str):
        kp_id = body_kp_name_to_id[kp_str]
        assert kp_id >= 0 and kp_id < 18
        kp_id *= 3
        if self.array_55[kp_id + 2] < 0:
            return None
        else:
            return (self.array_55[kp_id].astype(np.int32), self.array_55[kp_id + 1].astype(np.int32)) #, self.array_55[kp_id + 2])

class Poses:
    def __init__(self, array_n_55):
        """
        array_n_55 is a 2d array with n is number of persons detected 
        and 55 for the full description of a pose
        """
        self.poses = [ Pose(a) for a in array_n_55 ]
        self.array_n_55 = array_n_55

    def best(self):
        """
        When there are several persons detected, we want the one with the best confidence
        """
        if self.array_n_55.shape[0] == 0: # No person detected
            return None
        else:
            id_best = np.argmax(self.array_n_55[:,54])
            return self.poses[id_best]