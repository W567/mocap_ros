import numpy as np
import rospy
import tf
from math import sqrt
from copy import deepcopy


# Function to calculate Euclidean distance between two points (using translation part of the transform)
def calculate_distance(frame1, frame2, listener):
    try:
        # Get the transformation from frame1 to frame2
        (trans, rot) = listener.lookupTransform(frame1, frame2, rospy.Time(0))

        # Euclidean distance between two points (translation vectors)
        distance = sqrt(trans[0] ** 2 + trans[1] ** 2 + trans[2] ** 2)
        return distance
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        return None


# Function to calculate distances for all finger joints
def calculate_distances(finger_frames, listener):
    distances = []
    for i in range(len(finger_frames) - 1):
        dist = calculate_distance(finger_frames[i], finger_frames[i + 1], listener)
        if dist is not None:
            distances.append(dist)
    return np.array(distances)


# Initialize the ROS node
rospy.init_node('tf_distance_calculator')

# Initialize the tf listener
listener = tf.TransformListener()

# Define the frame names for each finger
right_hand_frames = [
    ["right_hand/wrist", "right_hand/index0", "right_hand/index1", "right_hand/index2", "right_hand/index3"],
    ["right_hand/wrist", "right_hand/middle0", "right_hand/middle1", "right_hand/middle2", "right_hand/middle3"],
    ["right_hand/wrist", "right_hand/pinky0", "right_hand/pinky1", "right_hand/pinky2", "right_hand/pinky3"],
    ["right_hand/wrist", "right_hand/ring0", "right_hand/ring1", "right_hand/ring2", "right_hand/ring3"],
    ["right_hand/wrist", "right_hand/thumb0", "right_hand/thumb1", "right_hand/thumb2", "right_hand/thumb3"]
]

# Variables to store previous and current distance arrays
prev_distance_array = None
current_distance_array = None

# Loop to continuously calculate distances while the ROS system is running
while not rospy.is_shutdown():
    # Calculate distances for each finger
    all_distances = []
    for finger_frames in right_hand_frames:
        distances = calculate_distances(finger_frames, listener)
        all_distances.extend(distances)

    current_distance_array = np.array(all_distances)

    # If it's not the first timestep, calculate the difference between current and previous distances
    if prev_distance_array is not None:
        print(f"Current distances:\n"
              f"{current_distance_array.reshape(-1, 4)}")
        distance_diff = current_distance_array - prev_distance_array
        print(f"Distance differences:\n"
              f"{distance_diff.reshape(-1, 4)}")

    # Update previous distance array
    if current_distance_array is not None and len(current_distance_array) > 0:
        prev_distance_array = deepcopy(current_distance_array)

    # Sleep for the next timestep (adjust based on desired rate)
    rospy.sleep(0.1)  # 10Hz, adjust if needed
