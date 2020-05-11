from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import rospy

LIGHT_THRESHOLD = .3  # minimum score needed to classify light    

class TLClassifier(object):
    def __init__(self, is_site):
        if is_site:
            graph_file = r'light_classification/model/ssd_site/frozen_inference_graph.pb' 
        else:
            graph_file = r'light_classification/model/ssd_sim/frozen_inference_graph.pb' 

        self.graph = tf.Graph()

        with self.graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as gfile:
                graph_def.ParseFromString(gfile.read())
                tf.import_graph_def(graph_def, name='')
            
            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

        self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.graph.as_default():
            image_expanded = np.expand_dims(image, axis=0)
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict = {self.image_tensor: image_expanded}
                )
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

   #     rospy.logwarn('Scores {0}'.format(scores))
    #    rospy.logwarn('classes {0}'.format(classes))

        if scores[0] > LIGHT_THRESHOLD:
            if classes[0] == 2:
                return TrafficLight.RED, scores[0]
            elif classes[0] == 3:
                return TrafficLight.YELLOW, scores[0]
            elif classes[0] == 1:
                return TrafficLight.GREEN, scores[0]                

        return TrafficLight.UNKNOWN, scores[0]
