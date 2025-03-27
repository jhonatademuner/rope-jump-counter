import cv2
import mediapipe as mp

class PoseDetector():
    """
    A class for detecting and drawing human pose landmarks using MediaPipe.
    """
    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        # Initialize parameters for pose detection and tracking
        self.mode = mode                        # Whether to treat the input images as static
        self.smooth = smooth                    # Whether to smooth the landmarks across frames
        self.detectionCon = detectionCon        # Minimum confidence value for detection
        self.trackCon = trackCon                # Minimum confidence value for tracking
        self.pTime = 0                          # Previous time used for FPS calculation (if needed)

        # Initialize MediaPipe drawing utilities and pose model
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def find_pose(self, img, draw=True):
        """
        Process the image to detect the pose, and optionally draw the landmarks.
        """
        # Convert the BGR image to RGB as MediaPipe expects RGB images
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the RGB image to find pose landmarks
        self.results = self.pose.process(imgRGB)

        # If landmarks are detected and drawing is enabled, draw them on the original image
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def get_points_positions(self, img):
        """
        Extract the positions (pixel coordinates) of the detected landmarks.
        """
        self.lmList = []  # List to store landmark positions
        # If landmarks are detected, iterate through each landmark
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape  # Get the image dimensions
                cx, cy = int(lm.x * w), int(lm.y * h)  # Convert normalized coordinates to pixel coordinates
                self.lmList.append([id, cx, cy])  # Append the landmark id and coordinates
        return self.lmList

    def draw_custom_points(self, image, landmark_list):
        """
        Draws custom points on the image based on specified landmark indices.
        """
        # Define sets of landmark indices for different body parts
        face_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        left_shoulder_indices = [12]
        right_shoulder_indices = [11]
        left_elbow_indices = [14]
        right_elbow_indices = [13]
        left_hand_indices = [22, 20, 18, 16]
        right_hand_indices = [21, 19, 17, 15]
        left_waist_indices = [24]
        right_waist_indices = [23]
        left_kne_indicese = [26]
        right_knee_indices = [25]
        left_foot_indices = {27, 29, 31}
        right_foot_indices = {28, 30, 32}

        # Loop through each landmark in the list
        for idx, x, y in landmark_list:
            # If the landmark is part of the left foot, draw a green circle
            if idx in left_foot_indices:
                cv2.circle(image, (x, y), 7, (0, 255, 0), cv2.FILLED)
            # If the landmark is part of the right foot, draw a blue circle
            elif idx in right_foot_indices:
                cv2.circle(image, (x, y), 7, (255, 0, 0), cv2.FILLED)
            # Otherwise, draw a white circle for other landmarks
            else:
                cv2.circle(image, (x, y), 5, (255, 255, 255), cv2.FILLED)
        return image

def calculate_center(points):
    """
    Calculates the center (average) position of the given points.
    Returns a tuple representing the (x, y) center.
    """
    if not points:
        return (0, 0)
    x_sum = sum(point[1] for point in points)
    y_sum = sum(point[2] for point in points)
    return (x_sum / len(points), y_sum / len(points))

def get_left_foot_position(landmark_list):
    """
    Returns the center position of the left foot using landmarks indices 26, 28, and 30.
    """
    try:
        # Get points corresponding to left foot landmarks and calculate center
        points = [landmark_list[26], landmark_list[28], landmark_list[30]]
        return calculate_center(points)
    except IndexError:
        # Return default if landmarks are missing
        return (0, 0)

def get_right_foot_position(landmark_list):
    """
    Returns the center position of the right foot using landmarks indices 27, 29, and 31.
    """
    try:
        # Get points corresponding to right foot landmarks and calculate center
        points = [landmark_list[27], landmark_list[29], landmark_list[31]]
        return calculate_center(points)
    except IndexError:
        # Return default if landmarks are missing
        return (0, 0)

def is_point_above(point1, point2, threshold=0):
    """
    Checks if point1 is above point2 by at least the threshold value.
    Uses the y-coordinate (smaller y value indicates higher position).
    """
    return (point1[1] - point2[1]) < -threshold

def is_point_below(point1, point2, threshold=0):
    """
    Checks if point1 is below point2 by at least the threshold value.
    """
    return (point1[1] - point2[1]) > threshold

def main(): 
    # Initialize the pose detector with default settings
    detector = PoseDetector()
    # Define the path to the video file
    video_path = 'rope_jumping_sample.mp4'
    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    frame_count = 0                      # Counter for frames processed
    start_time = cv2.getTickCount()      # Start time for elapsed time calculation
    jumps = 0                            # Counter for detected jumps
    left_foot_prev = (0, 0)              # Previous left foot position
    right_foot_prev = (0, 0)             # Previous right foot position
    left_foot_direction = 'neutral'      # Current movement direction for left foot
    right_foot_direction = 'neutral'     # Current movement direction for right foot

    # Set a threshold based on the frame height to determine significant movement
    threshold = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) / 100

    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if not success:
            break  # Exit loop if there are no more frames

        # Calculate elapsed time in seconds from the start
        elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        frame_count += 1  # Increment frame counter

        # Display frame count, elapsed time, and jump count on the frame
        cv2.putText(frame, f'Frame: {frame_count}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.putText(frame, f'Time: {elapsed_time:.2f}s', (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.putText(frame, f'Jumps: {jumps}', (10, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        # Process the frame to detect the pose and draw landmarks
        frame = detector.find_pose(frame)
        landmark_list = detector.get_points_positions(frame)
        frame = detector.draw_custom_points(frame, landmark_list)

        # For the very first frame, initialize previous foot positions
        if frame_count == 1:
            left_foot_prev = get_left_foot_position(landmark_list)
            right_foot_prev = get_right_foot_position(landmark_list)

        # Process every 5 frames to check for jumping motion
        if frame_count % 5 == 0:
            # Get the current positions of the left and right foot
            left_foot_curr = get_left_foot_position(landmark_list)
            right_foot_curr = get_right_foot_position(landmark_list)

            is_jumping = False  # Flag to determine if a jump is detected

            # Check if left foot has moved upward significantly
            if is_point_above(left_foot_curr, left_foot_prev, threshold):
                if left_foot_direction == 'neutral' or left_foot_direction == 'down':
                    is_jumping = True
                left_foot_direction = 'up'
            # Check if left foot has moved downward significantly
            elif is_point_below(left_foot_curr, left_foot_prev, threshold):
                left_foot_direction = 'down'
            else:
                left_foot_direction = 'neutral'

            # Check if right foot has moved upward significantly
            if is_point_above(right_foot_curr, right_foot_prev, threshold):
                if right_foot_direction == 'neutral' or right_foot_direction == 'down':
                    is_jumping = True
                right_foot_direction = 'up'
            # Check if right foot has moved downward significantly
            elif is_point_below(right_foot_curr, right_foot_prev, threshold):
                right_foot_direction = 'down'
            else:
                right_foot_direction = 'neutral'

            # If a jump is detected, increment the jump counter
            if is_jumping:
                jumps += 1

            # Update previous foot positions for the next check
            left_foot_prev = left_foot_curr
            right_foot_prev = right_foot_curr

        # Display the processed frame in a window named "Pose Detection"
        cv2.imshow("Pose Detection", frame)
        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    # Print the total number of jumps detected
    print(f'Total jumps: {jumps}')

# Run the main function if this script is executed
if __name__ == "__main__":
    main()
