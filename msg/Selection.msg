# Bounding boxes in pixels
sensor_msgs/RegionOfInterest box

# Integer class IDs for each bounding box
int32 class_id

# String class IDs for each bouding box
string class_name

# Float probability scores of the class_id
float32 score

# Instance masks as Image
geometry_msgs/Point centroid

# Current camera link frame relative to base link
geometry_msgs/TransformStamped transform