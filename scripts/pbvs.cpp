/**
 * This node takes in desired camera pose and current camera pose and uses VISP code to
 * calculate the end effector velocity needed to minimize the error between two poses mentioned.
 */

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <thread>

#include <kortex_driver/ActionNotification.h>
#include <kortex_driver/ActionEvent.h>
#include <kortex_driver/Base_ClearFaults.h>
#include <kortex_driver/ReadAction.h>
#include <kortex_driver/ExecuteAction.h>
#include <kortex_driver/OnNotificationActionTopic.h>
#include <kortex_driver/SendGripperCommand.h>
#include <kortex_driver/GripperMode.h>

// ViSP
#include <visp3/core/vpMath.h>
#include <visp3/core/vpColVector.h>
#include <visp3/core/vpHomogeneousMatrix.h>
#include <visp3/core/vpMatrix.h>
#include <visp3/core/vpTranslationVector.h>
#include <visp3/core/vpQuaternionVector.h>
#include <visp3/visual_features/vpFeatureThetaU.h>
#include <visp3/visual_features/vpFeatureTranslation.h>
#include <visp3/vs/vpServo.h>

#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <raf/CartVelCmd.h>
#include <raf/RafState.h>
#include <raf/FaceDetection.h>

#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#define HOME_ACTION_IDENTIFIER 2
#define PI 3.14159265

class VisualServoing{
    private:
        ros::NodeHandle nh;
        ros::Subscriber sub_notif;
        ros::Subscriber sub_state;
        ros::Subscriber sub_face_detection;
        ros::Publisher cart_vel_pub;
        ros::Publisher twist_pub;

        tf2_ros::Buffer tf_buffer;
        tf2_ros::TransformListener* tf_listener;
        geometry_msgs::TransformStamped tf_transform;

        //parameters
        double pbvs_control_loop_hz;
        double pbvs_control_law_gain_lambda;
        double pbvs_control_deadband_error;

        double xyz_vel_limit;
        double rpy_vel_limit;

        std::string desired_camera_frame;
        std::string current_camera_frame;
        std::string robot_base_frame;
        std::string control_input_topic;

        std::string blah;

        raf::CartVelCmd control_input;
        kortex_driver::TwistCommand twist_control_input;
        raf::RafState raf_state;
        raf::FaceDetection face_detection;
        // bool enable_arm_detections = false;
        // bool enable_scene_detections = false;
        // bool enable_face_detections = false;
        // bool visualize_face_detections = false;
        // std::string visualize_detections = "default";
        // std::string view = "default";
        // std::string system_state = "default";

        std::atomic<int> last_action_notification_event{0};
        std::string robot_name = "my_gen3";
        bool success = true;
        int degrees_of_freedom = 6;
        bool is_gripper_present = false;

    public:
        VisualServoing(){
            // Subscribers
            sub_notif = nh.subscribe("/" + robot_name  + "/action_topic", 1000, &VisualServoing::notification_callback, this);
            sub_state = nh.subscribe("/raf_state", 1000, &VisualServoing::state_callback, this);
            sub_face_detection = nh.subscribe("/face_detection", 1, &VisualServoing::face_detection_callback, this);

            // Publishers
            cart_vel_pub = nh.advertise<raf::CartVelCmd>(control_input_topic, 1);
            twist_pub = nh.advertise<kortex_driver::TwistCommand>("my_gen3/in/cartesian_velocity", 1);

            // Transform Listener
            tf_listener = new tf2_ros::TransformListener(tf_buffer);

            VisualServoing::initialize();
            VisualServoing::pbvs();
        }

        void callback(const std_msgs::String::ConstPtr& msg){
            ROS_WARN("I heard: [%s]", msg->data.c_str());
        }

        void face_detection_callback(const raf::FaceDetection& msg){
            VisualServoing::face_detection = msg;
        }

        void notification_callback(const kortex_driver::ActionNotification& notif){
            last_action_notification_event = notif.action_event;
        }

        void state_callback(const raf::RafState& msg){
            VisualServoing::raf_state = msg;
            // VisualServoing::enable_arm_detections = msg.enable_arm_detections;
            // VisualServoing::enable_scene_detections = msg.enable_scene_detections;
            // VisualServoing::enable_face_detections = msg.enable_face_detections;
            // VisualServoing::visualize_face_detections = msg.visualize_face_detections;
            // VisualServoing::visualize_detections = msg.visualize_detections;
            // VisualServoing::view = msg.view;
            // VisualServoing::system_state = msg.system_state;
            // ROS_INFO()"RAF State (enable face detections): [%s]", std::to_string(raf_state.enable_face_detections).c_str());
        }

        void initialize(){
            // Parameter robot_name
            if (!ros::param::get("~robot_name", robot_name))
            {
                std::string error_string = "Parameter robot_name was not specified, defaulting to " + robot_name + " as namespace";
                ROS_WARN("%s", error_string.c_str());
            }
            else 
            {
                std::string error_string = "Using robot_name " + robot_name + " as namespace";
                ROS_INFO("%s", error_string.c_str());
            }

            // Parameter degrees_of_freedom
            if (!ros::param::get("/" + robot_name + "/degrees_of_freedom", degrees_of_freedom))
            {
                std::string error_string = "Parameter /" + robot_name + "/degrees_of_freedom was not specified, defaulting to " + std::to_string(degrees_of_freedom) + " as degrees of freedom";
                ROS_WARN("%s", error_string.c_str());
            }
            else 
            {
                std::string error_string = "Using degrees_of_freedom " + std::to_string(degrees_of_freedom) + " as degrees_of_freedom";
                ROS_INFO("%s", error_string.c_str());
            }

            // Parameter is_gripper_present
            if (!ros::param::get("/" + robot_name + "/is_gripper_present", is_gripper_present))
            {
                std::string error_string = "Parameter /" + robot_name + "/is_gripper_present was not specified, defaulting to " + std::to_string(is_gripper_present);
                ROS_WARN("%s", error_string.c_str());
            }
            else 
            {
                std::string error_string = "Using is_gripper_present " + std::to_string(is_gripper_present);
                ROS_INFO("%s", error_string.c_str());
            }

            // We need to call this service to activate the Action Notification on the kortex_driver node.
            ros::ServiceClient service_client_activate_notif = nh.serviceClient<kortex_driver::OnNotificationActionTopic>("/" + robot_name + "/base/activate_publishing_of_action_topic");
            kortex_driver::OnNotificationActionTopic service_activate_notif;
            if (service_client_activate_notif.call(service_activate_notif))
            {
                ROS_INFO("Action notification activated!");
            }
            else 
            {
                std::string error_string = "Action notification publication failed";
                ROS_ERROR("%s", error_string.c_str());
                success = false;
            }

            nh.getParam("/pbvs/pbvs_control_loop_hz", pbvs_control_loop_hz);
            nh.getParam("/pbvs/pbvs_control_law_gain_lambda", pbvs_control_law_gain_lambda);
            nh.getParam("/pbvs/pbvs_control_deadband_error", pbvs_control_deadband_error);

            nh.getParam("/pbvs/xyz_vel_limit", xyz_vel_limit);
            nh.getParam("/pbvs/rpy_vel_limit", rpy_vel_limit);

            nh.getParam("/pbvs/desired_camera_frame", desired_camera_frame);
            nh.getParam("/pbvs/current_camera_frame", current_camera_frame);
            nh.getParam("/pbvs/robot_base_frame", robot_base_frame);
            nh.getParam("/pbvs/control_input_topic", control_input_topic);

            std::string info = "\nPBVS Parameters: ";
            ROS_INFO("%s", info.c_str());
            ROS_INFO("Control Loop (Hz) = %s", std::to_string(pbvs_control_loop_hz).c_str());
            ROS_INFO("Control Law Gain (lambda) = %s", std::to_string(pbvs_control_law_gain_lambda).c_str());
            ROS_INFO("Control Deadband Error (m) = %s", std::to_string(pbvs_control_deadband_error).c_str());
            ROS_INFO("Cartesian Velocity Limit = %s", std::to_string(xyz_vel_limit).c_str());
            ROS_INFO("Rotational Velocity Limit = %s", std::to_string(rpy_vel_limit).c_str());
        }

        bool wait_for_action_end_or_abort(){
            while (ros::ok())
            {
                if (last_action_notification_event.load() == kortex_driver::ActionEvent::ACTION_END)
                {
                ROS_INFO("Received ACTION_END notification");
                return true;
                }
                else if (last_action_notification_event.load() == kortex_driver::ActionEvent::ACTION_ABORT)
                {
                ROS_INFO("Received ACTION_ABORT notification");
                return false;
                }
                ros::spinOnce();
            }
            return false;
        }

        bool clear_faults(ros::NodeHandle n, const std::string &robot_name){
            ros::ServiceClient service_client_clear_faults = n.serviceClient<kortex_driver::Base_ClearFaults>("/" + robot_name + "/base/clear_faults");
            kortex_driver::Base_ClearFaults service_clear_faults;

            // Clear the faults
            if (!service_client_clear_faults.call(service_clear_faults))
            {
                std::string error_string = "Failed to clear the faults";
                ROS_ERROR("%s", error_string.c_str());
                return false;
            }

            // Wait a bit
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            return true;
        }

        bool home_the_robot(ros::NodeHandle n, const std::string &robot_name){
            ros::ServiceClient service_client_read_action = n.serviceClient<kortex_driver::ReadAction>("/" + robot_name + "/base/read_action");
            kortex_driver::ReadAction service_read_action;
            last_action_notification_event = 0;

            // The Home Action is used to home the robot. It cannot be deleted and is always ID #2:
            service_read_action.request.input.identifier = HOME_ACTION_IDENTIFIER;

            if (!service_client_read_action.call(service_read_action))
            {
                std::string error_string = "Failed to call ReadAction";
                ROS_ERROR("%s", error_string.c_str());
                return false;
            }

            // We can now execute the Action that we read 
            ros::ServiceClient service_client_execute_action = n.serviceClient<kortex_driver::ExecuteAction>("/" + robot_name + "/base/execute_action");
            kortex_driver::ExecuteAction service_execute_action;

            service_execute_action.request.input = service_read_action.response.output;
            
            if (service_client_execute_action.call(service_execute_action))
            {
                ROS_INFO("The Home position action was sent to the robot.");
            }
            else
            {
                std::string error_string = "Failed to call ExecuteAction";
                ROS_ERROR("%s", error_string.c_str());
                return false;
            }

            return wait_for_action_end_or_abort();
        }

        bool send_gripper_command(ros::NodeHandle n, const std::string &robot_name, double value){
            // Initialize the ServiceClient
            ros::ServiceClient service_client_send_gripper_command = n.serviceClient<kortex_driver::SendGripperCommand>("/" + robot_name + "/base/send_gripper_command");
            kortex_driver::SendGripperCommand service_send_gripper_command;

            // Initialize the request
            kortex_driver::Finger finger;
            finger.finger_identifier = 0;
            finger.value = value;
            service_send_gripper_command.request.input.gripper.finger.push_back(finger);
            service_send_gripper_command.request.input.mode = kortex_driver::GripperMode::GRIPPER_POSITION;

            if (service_client_send_gripper_command.call(service_send_gripper_command))  
            {
                ROS_INFO("The gripper command was sent to the robot.");
            }
            else
            {
                std::string error_string = "Failed to call SendGripperCommand";
                ROS_ERROR("%s", error_string.c_str());
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            return true;
        }

        void getTwistVectorBodyFrame(Eigen::VectorXd& Vb, Eigen::VectorXd Vc, Eigen::Matrix4d bMc){
            // calculate adjoint using input transformation
            // [  R  0]
            // [[tR] R]
            Eigen::Matrix3d bRc = bMc.block<3,3>(0,0); // rotation
            Eigen::Vector3d btc = bMc.block<3,1>(0,3); // translation

            // skew symmetric [t]
            Eigen::Matrix3d btc_s;
            btc_s << 0, -btc(2), btc(1),
                    btc(2), 0, -btc(0),
                    -btc(1), btc(0), 0;

            // Adjoint
            Eigen::MatrixXd bAdc(6, 6);
            bAdc << bRc, Eigen::Matrix3d::Zero(), 
                    btc_s*bRc, bRc;

            // calculate twist in body frame using adjoint and twist in camera frame
            Vb = bAdc * Vc;

            // std::stringstream ss;
            // ss.str(std::string());
            // ss << "Vb:\n" << Vb << "\n";
            // ROS_INFO("\n%s", ss.str().c_str());
        }

        void fixQuat(double &qx, double &qy, double &qz, double &qw){
            double norm = sqrt(qx*qx + qy*qy + qz*qz + qw*qw);
            qx = qx/norm;
            qy = qy/norm;
            qz = qz/norm;
            qw = qw/norm;

            if(2*acos(qw) > PI)
            {
                qx = -qx;
                qy = -qy;
                qz = -qz;
                qw = -qw;
            }
        }

        void limitLinVel(double &v){
            v = std::min(v, xyz_vel_limit);
            v = std::max(v, -xyz_vel_limit);
        }

        void limitRotVel(double &w){
            w = std::min(w, rpy_vel_limit);
            w = std::max(w, -rpy_vel_limit);
        }

        void pbvs(){
            std::string info_msg;
            // ROS_INFO("%s", info_msg.c_str());

            // success &= clear_faults(nh, robot_name);
            // // success &= home_the_robot(nh, robot_name);
            // if (is_gripper_present){
            //     success &= send_gripper_command(nh, robot_name, 0.95);
            //     ros::Duration(0.5).sleep();
            //     success &= send_gripper_command(nh, robot_name, 0.0);
            // }

            // std::string enter;
            // std::cout << "\n **** Type ENTER to start visual servoing... **** \n"; std::cin >> enter;

            info_msg = "\nStarting Visual Servoing...";
            ROS_INFO("%s", info_msg.c_str());

            vpHomogeneousMatrix cdMc; // cdMc is the result of a pose estimation; cd: desired camera frame, c:current camera frame

            // Creation of the current visual feature s = (c*_t_c, ThetaU)
            vpFeatureTranslation s_t(vpFeatureTranslation::cdMc);
            vpFeatureThetaU s_tu(vpFeatureThetaU::cdRc);

            // Set the initial values of the current visual feature s = (c*_t_c, ThetaU)
            s_t.buildFrom(cdMc);
            s_tu.buildFrom(cdMc);

            // Build the desired visual feature s* = (0,0)
            vpFeatureTranslation s_star_t(vpFeatureTranslation::cdMc); // Default initialization to zero
            vpFeatureThetaU s_star_tu(vpFeatureThetaU::cdRc); // Default initialization to zero
            vpColVector v; // Camera velocity
            double error = 1.0;  // Task error

            // Creation of the visual servo task.
            vpServo task;
            // Visual servo task initialization
            // - Camera is monted on the robot end-effector and velocities are
            //   computed in the camera frame
            task.setServo(vpServo::EYEINHAND_CAMERA);
            // - Interaction matrix is computed with the current visual features s
            task.setInteractionMatrixType(vpServo::MEAN);

            // vpAdaptiveGain lambda;
            // lambda.initStandard(4, 0.4, 30);

            // - Set the contant gain to 1
            task.setLambda(pbvs_control_law_gain_lambda);
            // - Add current and desired translation feature
            task.addFeature(s_t, s_star_t);
            // - Add current and desired ThetaU feature for the rotation
            task.addFeature(s_tu, s_star_tu);
            // Visual servoing loop. The objective is here to update the visual
            // features s = (c*_t_c, ThetaU), compute the control law and apply
            // it to the robot

            ros::Rate rate(pbvs_control_loop_hz);
            while(nh.ok()){

                if (VisualServoing::raf_state.enable_visual_servoing == false) {
                    ros::spinOnce();
                    continue;
                }

                // Only enable visual servoing if the mouth is closed
                if (VisualServoing::face_detection.mouth_open == true) {
                    control_input.velocity.data = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                    twist_control_input.twist.linear_x = 0.0;
                    twist_control_input.twist.linear_y = 0.0;
                    twist_control_input.twist.linear_z = 0.0;
                    twist_control_input.twist.angular_x = 0.0;
                    twist_control_input.twist.angular_y = 0.0;
                    twist_control_input.twist.angular_z = 0.0;
                    cart_vel_pub.publish(control_input);
                    twist_pub.publish(twist_control_input);
                    ros::spinOnce();
                    continue;
                }

                try{
                    // lookup desired camera from and current camera frame transform
                    tf_transform = tf_buffer.lookupTransform(desired_camera_frame, current_camera_frame, ros::Time(0), ros::Duration(0.5));

                    // convert transform to vpHomogeneousMatrix
                    double t_x = tf_transform.transform.translation.x;
                    double t_y = tf_transform.transform.translation.y;
                    double t_z = tf_transform.transform.translation.z;
                    vpTranslationVector trans_vec;
                    trans_vec.buildFrom(t_x, t_y, t_z);

                    double q_x = tf_transform.transform.rotation.x;
                    double q_y = tf_transform.transform.rotation.y;
                    double q_z = tf_transform.transform.rotation.z;
                    double q_w = tf_transform.transform.rotation.w;
                    fixQuat(q_x, q_y, q_z, q_w);
                    vpQuaternionVector quat_vec;
                    quat_vec.buildFrom(q_x, q_y, q_z, q_w);

                    // vpHomogeneousMatrix cdMc;
                    cdMc.buildFrom(trans_vec, quat_vec);

                    // PBVS
                    // ... cdMc is here the result of a pose estimation
                    // Update the current visual feature s
                    s_t.buildFrom(cdMc);  // Update translation visual feature
                    s_tu.buildFrom(cdMc); // Update ThetaU visual feature
                    v = task.computeControlLaw(); // Compute camera velocity skew
                    error = (task.getError()).sumSquare(); // error = s^2 - s_star^2

                    // convert twist in camera frame to body frame
                    // rearranging twist from [v w] to [w v]
                    Eigen::VectorXd Vc(6);
                    Vc << v[3], v[4], v[5], v[0], v[1], v[2];

                    // lookup desired camera from and robot body frame transform
                    Eigen::VectorXd Vb(6);
                    tf_transform = tf_buffer.lookupTransform(robot_base_frame, current_camera_frame, ros::Time(0), ros::Duration(3.0));
                    getTwistVectorBodyFrame(Vb, Vc, tf2::transformToEigen(tf_transform).matrix());

                    // limit linear and rotational velocity
                    limitLinVel(Vb[3]);
                    limitLinVel(Vb[4]);
                    limitLinVel(Vb[5]);
                    limitRotVel(Vb[0]);
                    limitRotVel(Vb[1]);
                    limitRotVel(Vb[2]);

                    // command end effector twist to robot
                    if(error >= pbvs_control_deadband_error){
                        control_input.velocity.data = {
                            static_cast<float>(Vb[3]), static_cast<float>(Vb[4]), static_cast<float>(Vb[5]), 
                            static_cast<float>(Vb[0]), static_cast<float>(Vb[1]), static_cast<float>(Vb[2])
                        };

                        twist_control_input.twist.linear_x = static_cast<float>(Vb[3]);
                        twist_control_input.twist.linear_y = static_cast<float>(Vb[4]);
                        twist_control_input.twist.linear_z = static_cast<float>(Vb[5]);
                        twist_control_input.twist.angular_z = static_cast<float>(Vb[0]);
                        twist_control_input.twist.angular_x = static_cast<float>(Vb[1]);
                        twist_control_input.twist.angular_y = static_cast<float>(Vb[2]);
                        // twist_control_input.twist.angular_x = 0.0;
                        // twist_control_input.twist.angular_y = 0.0;
                        // twist_control_input.twist.angular_z = 0.0;
                    }
                    else{
                        control_input.velocity.data = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                        twist_control_input.twist.linear_x = 0.0;
                        twist_control_input.twist.linear_y = 0.0;
                        twist_control_input.twist.linear_z = 0.0;
                        twist_control_input.twist.angular_x = 0.0;
                        twist_control_input.twist.angular_y = 0.0;
                        twist_control_input.twist.angular_z = 0.0;
                    }
                }
                catch(tf2::TransformException ex){
                    ROS_ERROR("%s", ex.what());
                    control_input.velocity.data = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                    twist_control_input.twist.linear_x = 0.0;
                    twist_control_input.twist.linear_y = 0.0;
                    twist_control_input.twist.linear_z = 0.0;
                    twist_control_input.twist.angular_x = 0.0;
                    twist_control_input.twist.angular_y = 0.0;
                    twist_control_input.twist.angular_z = 0.0;
                }

                cart_vel_pub.publish(control_input);
                twist_pub.publish(twist_control_input);

                // Log some data
                // std::stringstream ss0;
                // ss0.str(std::string());
                // ss0 << "body frame v[v w]:\n";
                // ss0 << control_input.velocity.data[0] << " \n";
                // ss0 << control_input.velocity.data[1] << " \n";
                // ss0 << control_input.velocity.data[2] << " \n";
                // ss0 << control_input.velocity.data[3] << " \n";
                // ss0 << control_input.velocity.data[4] << " \n";
                // ss0 << control_input.velocity.data[5] << " \n";
                // ss0 << "\n";
                // ROS_INFO("\n%s", ss0.str().c_str());

                // std::cout << std::boolalpha;
                // std::cout << "\nRAF State: \n";
                // std::cout << "Enable Arm Detections: " << std::to_string(VisualServoing::raf_state.enable_arm_detections) << " \n";
                // std::cout << "Enable Scene Detections: " << std::to_string(VisualServoing::raf_state.enable_scene_detections) << " \n";
                // std::cout << "Enable Face Detections: " << std::to_string(VisualServoing::raf_state.enable_face_detections) << " \n";
                // std::cout << "Visualize Face Detections: " << std::to_string(VisualServoing::raf_state.visualize_face_detections) << " \n";
                // std::cout << "Visualize Arm Detections: " << VisualServoing::raf_state.visualize_detections << " \n";
                // std::cout << "View: " << VisualServoing::raf_state.view << " \n";
                // std::cout << "System State: " << VisualServoing::raf_state.system_state << " \n";

                // std::cout << "\nFace Detection: \n";
                // std::cout << "Number of Faces: " << std::to_string(VisualServoing::face_detection.num_face) << " \n";
                // std::cout << "Mouth Open: " << std::to_string(VisualServoing::face_detection.mouth_open) << " \n";
                // std::cout << "Mouth X: " << std::to_string(VisualServoing::face_detection.mouth_x) << " \n";
                // std::cout << "Mouth Y: " << std::to_string(VisualServoing::face_detection.mouth_y) << " \n";
                // std::cout << "Ellipse a: " << std::to_string(VisualServoing::face_detection.a) << " \n";
                // std::cout << "Ellipse b: " << std::to_string(VisualServoing::face_detection.b) << " \n";
                // std::cout << "Ellipse theta: " << std::to_string(VisualServoing::face_detection.theta) << " \n";

                ros::spinOnce();
                rate.sleep();
            }
            task.kill();
        }
};

int main(int argc, char** argv){
    ros::init(argc, argv, "visual_servoing");
    VisualServoing vsObject;
    ros::spin();
    return 0;
}