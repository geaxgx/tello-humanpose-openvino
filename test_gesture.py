import argparse
from human_pose import *
from math import atan2, degrees, pi, sqrt
from modules.FPS import FPS


ap=argparse.ArgumentParser()
ap.add_argument("-i","--input",default="0",help="input video file (0, filename, rtsp://admin:admin@192.168.1.71/1, ...")
ap.add_argument('-d', '--device',
                        help='Specify the target device to infer on: CPU, GPU, FPGA, HDDL or MYRIAD. '
                            'The demo will look for a suitable plugin for device specified '
                            '(default=%(default)s)',
                        type=str, default='CPU')
args=ap.parse_args()

if args.input.isdigit():
    args.input=int(args.input)

# Read video
video=cv2.VideoCapture(args.input)
video.set(cv2.CAP_PROP_FRAME_WIDTH,640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
print("CAP_PROP_FRAME_WIDTH:",video.get(cv2.CAP_PROP_FRAME_WIDTH))
print("CAP_PROP_FRAME_HEIGHT:",video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("CAM FPS:",video.get(cv2.CAP_PROP_FPS))

my_hp = HumanPose(device=args.device, model_input_height=256)



def ccw(A,B,C):
    """
        Returns True if the 3 points A,B and C are listed in a counterclockwise order 
        ie if the slope of the line AB is less than the slope of AC
        https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
    """
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def intersect(A,B,C,D):
    """
        Return true if line segments AB and CD intersect
        https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
    """
    if A is None or B is None or C is None or D is None:
        return False
    else:
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
    
def angle (A, B, C):
    """
        Calculate the angle between segment(A,B) and segment (B,C)
    """
    if A is None or B is None or C is None:
        return None
    return degrees(atan2(C[1]-B[1],C[0]-B[0]) - atan2(A[1]-B[1],A[0]-B[0]))%360

def vertical_angle (A, B):
    """
        Calculate the angle between segment(A,B) and vertical axe
    """
    if A is None or B is None:
        return None
    return degrees(atan2(B[1]-A[1],B[0]-A[0]) - pi/2)

def sq_distance (A, B):
    """
        Calculate the square of the distance between points A and B
    """
    return (B[0]-A[0])**2 + (B[0]-A[0])**2

def distance (A, B):
    """
        Calculate the square of the distance between points A and B
    """
    return sqrt((B[0]-A[0])**2 + (B[1]-A[1])**2)



def check_pose(skel):
    """
        kps: keypoints of one person. Shape = (25,3)
    """
    neck = skel.get_body_kp("Neck")
    
    r_wrist = skel.get_body_kp("RWrist")
    l_wrist = skel.get_body_kp("LWrist")
    r_elbow = skel.get_body_kp("RElbow")
    l_elbow = skel.get_body_kp("LElbow")
    r_shoulder = skel.get_body_kp("RShoulder")
    l_shoulder = skel.get_body_kp("LShoulder")
    r_ear = skel.get_body_kp("REar")
    l_ear = skel.get_body_kp("LEar") 

    shoulders_width = distance(r_shoulder,l_shoulder) if r_shoulder and l_shoulder else None

    vert_angle_right_arm = vertical_angle(r_wrist, r_elbow)
    vert_angle_left_arm = vertical_angle(l_wrist, l_elbow)

    left_hand_up = neck and l_wrist and l_wrist[1] < neck[1]
    right_hand_up = neck and r_wrist and r_wrist[1] < neck[1]

    if right_hand_up:
        if not left_hand_up:
            # Only right arm up
            if r_ear and (r_ear[0]-neck[0])*(r_wrist[0]-neck[0])>0:
                # Right ear and right hand on the same side
                if vert_angle_right_arm:
                    if vert_angle_right_arm < -15:
                        return "RIGHT_ARM_UP_OPEN"
                    if 15 < vert_angle_right_arm < 90:
                        return "RIGHT_ARM_UP_CLOSED"
            elif l_ear and r_wrist[1]>l_ear[1] and shoulders_width : #and distance(r_wrist,l_ear) < shoulders_width/4:
                # Right hand close to left ear
                return "RIGHT_HAND_ON_LEFT_EAR"

        else:
            # Both hands up
            # Check if both hands are on the ears
          
            # if r_ear and l_ear:
            #     ear_dist = distance(r_ear,l_ear)
            #     if distance(r_wrist,r_ear)<ear_dist/3 and distance(l_wrist,l_ear)<ear_dist/3:
            #         return("HANDS_ON_EARS")
            # Check if boths hands are closed to each other and above ears 
            # (check right hand is above right ear is enough since hands are closed to each other)
            if shoulders_width and r_ear:
                near_dist = shoulders_width
                if r_ear[1] > r_wrist[1] and distance(r_wrist, l_wrist) < near_dist :
                    return "CLOSE_HANDS_UP"




    else:
        if left_hand_up:
            # Only left arm up
            if l_ear and (l_ear[0]-neck[0])*(l_wrist[0]-neck[0])>0:
                # Left ear and left hand on the same side
                if vert_angle_left_arm:
                    if vert_angle_left_arm < -15:
                        return "LEFT_ARM_UP_CLOSED"
                    if 15 < vert_angle_left_arm < 90:
                        return "LEFT_ARM_UP_OPEN"
            elif r_ear and l_wrist[1]>r_ear[1] and shoulders_width: # and distance(l_wrist,r_ear) < shoulders_width/4:
                # Left hand close to right ear
                return "LEFT_HAND_ON_RIGHT_EAR"

        else:
            # Both wrists under the neck
            if neck and shoulders_width and r_wrist and l_wrist:
                near_dist = shoulders_width/3
                if distance(r_wrist, neck) < near_dist and distance(l_wrist, neck) < near_dist :
                    return "HANDS_ON_NECK"



    return None

fps = FPS()
while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break
    # nb_persons, kps,_ = my_hp.eval(frame)
    poses_2d = my_hp.eval(frame)
    skel = Poses(poses_2d).best()
    my_hp.draw(frame, poses_2d)
    if skel is not None:
        pose = check_pose(skel)
        if pose: 
            print(pose)
            cv2.putText(frame, pose,(20, 80), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255),2)
    fps.update()
    fps.display(frame)
    cv2.imshow("Rendering", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break
    elif k== 32: # space
        cv2.waitKey(0)
    elif k==48: #0
        print("kp",my_op.get_body_kp(0,"Neck"))
    

video.release()
cv2.destroyAllWindows()
