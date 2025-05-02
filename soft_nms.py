import numpy as np
import matplotlib.pyplot as plt
#TODO Go through Code usage and check if all functions are used
def circles_area_overlap(A, B):
    d = np.hypot(B['x'] - A['x'], B['y'] - A['y'])
    # no overlap if d >= A['r']+B['r'], then can skip calculations and have 0 overlap
    if d < A['r'] + B['r']:
        Arr = A['r'] ** 2
        Brr = B['r'] ** 2
        # Check if one cirle is inside the other, if so send smallest circle area as result
        if d <= abs(B['r'] - A['r']):
            return np.pi * min(Arr, Brr)

        # One angle for each circle sector
        tA = 2 * np.arccos((Arr + d**2 - Brr) / (2 * d * A['r']))
        tB = 2 * np.arccos((Brr + d**2 - Arr) / (2 * d * B['r']))

        # Two circle sector areas minus two triangle areas calculation
        return 0.5 * (Arr * (tA - np.sin(tA)) + Brr * (tB - np.sin(tB)))

    return 0

def draw_circles_on_tensor(tensor, radius=0.5):
    """
    Draw circles at all coordinates where the first channel is over 0.5.
    
    Parameters:
        tensor (torch.Tensor): A tensor with shape (M, N, 3), where the first channel
                               contains values to check for circles.
        radius (float): The radius of the circles to draw. Default is 0.5.
    """
    M, N, _ = tensor.shape
    plt.figure(figsize=(6, 6))
    for i in range(M):
        for j in range(N):
            if tensor[i, j, 0] > 0.5:
                circle = plt.Circle((j, i), radius, color='red', fill=False, linewidth=2)
                plt.gca().add_patch(circle)
    plt.xlim(0, N)
    plt.ylim(0, M)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.show()

def get_detected_circles_from_tensor(tensor, threshold_detection_score=0.5, radius_detection_circle=0.5, data=None):
    """
    Create a dictionary with coordinates and detection circle radius where the first channel is over a given threshold.
    The x and y values are taken from the second and third channels, respectively.
    
    Parameters:
        tensor (torch.Tensor): A tensor of shape (M, N, 3), where the second channel contains x values 
                               and the third channel contains y values.
        threshold_detection_score (float): The threshold for detection in the first channel. Default is 0.5.
        radius_detection_circle (float): The radius of the detection circle. Default is 0.5.
    
    Returns:
        dict: A dictionary with keys as indices (x, y) and values as the radius for each detection circle.
    """
    if data is None:
        detected_circles = {}
        counter = 0
    else:
        detected_circles = []
        id_counter = data[0]  
        use_loss = data[2]  
    M, N, _ = tensor.shape
    for i in range(M):
        for j in range(N):
            score = tensor[i, j, 0].item()  # Get the detection score from the first channel
            if score > threshold_detection_score:  # Check if the first channel is above threshold
                
                # Get the x and y coordinates from the second and third channels
                x_coord = tensor[i, j, 1].item() # Second channel (x-coordinate)
                y_coord = tensor[i, j, 2].item() # Third channel (y-coordinate)
                if use_loss[1]:
                    x_coord -= tensor[i, j, 3].item()
                    y_coord -= tensor[i, j, 4].item()
                
                # Add the detection info to the dictionary
                if data is None:
                    detected_circles[counter] = {
                        's': score,
                        'x': x_coord,
                        'y': y_coord,
                        'r': radius_detection_circle
                    }
                    # Increment the counter for the next detection
                    counter += 1
                else:
                    detected_circles.append({"id": id_counter,
                                  "keypoints": [[0,0,0], [x_coord, y_coord, 0]],
                                  "position_on_pitch": [x_coord, y_coord, 0],
                                  "score": score,
                                  "image_id": int(data[1]),
                                  "category_id": 1,
                                  "area": 0,
                                  'x': x_coord,
                                  'y': y_coord,
                                  "r": radius_detection_circle})
                    id_counter += 1

                
    if data is None:
        return detected_circles
    return detected_circles, id_counter

def plot_detected_circles(detected_circles, color='red'):
    """
    Plot circles on a 2D plot based on the detected circles dictionary.
    
    Parameters:
        detected_circles (dict): A dictionary where the key is the detection index and the value contains 'x', 'y' coordinates
                                  and 'radius_detection_circle' for plotting circles.
    """
    plt.figure(figsize=(6, 6))
    for detection in detected_circles:
        y = -detection['x']
        x = -detection['y']
        radius = detection['r']
        
        # Create and add the circle to the plot
        circle = plt.Circle((x, y), radius, color=color, fill=False, linewidth=2)
        plt.gca().add_patch(circle)

    # Get min and max values for x and y coordinates
    min_y = min(-detection['x'] for detection in detected_circles)
    max_y = max(-detection['x'] for detection in detected_circles)
    min_x = min(-detection['y'] for detection in detected_circles)
    max_x = max(-detection['y'] for detection in detected_circles)

    margin_x = (max_x-min_x) * 0.1
    margin_y = (max_y-min_y) * 0.1
    plt.xlim(min_x - margin_x, max_x + margin_x)
    plt.ylim(min_y - margin_y, max_y + margin_y)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.savefig(f'./data_illustration/output-{color}.png')

# Using sigmoid-hack for soft function
def overlap_based_weighting_function(IoU, w=20, b=-10):

    wxb = IoU * w + b
    return (1.0 - 1.0/(1.0+np.exp(-wxb)))

def soft_non_maximum_suppression(detected_circles, threshold_detection_score=0.5, w=20, b=-10):

    detections = []
    flag = 1
    i = 0
    while flag:
        max_detection = max(detected_circles, key=lambda x: x['score'])
        detections.append(max(detected_circles, key=lambda x: x['score']).copy())
        i += 1

        max_detection['score'] = 0.0
        ra = max_detection['r']
        A = np.pi * ra * ra

        flag = 0
        for detection in detected_circles:
            overlap_area = circles_area_overlap(detection, max_detection)
            rb = detection['r']
            B = np.pi * rb * rb
            IoU =  overlap_area / (A + B - overlap_area)

            if IoU > 0:
                f = overlap_based_weighting_function(IoU, w, b)
            else:
                f = 1.0
            detection['score'] = detection['score'] * f
            
            if flag == 0:
                # check if some detections remains (over detection threshold)
                if detection['score'] > threshold_detection_score:
                    flag = 1
    return detections

